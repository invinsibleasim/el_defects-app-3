
import os
import io
import cv2
import time
import json
import zipfile
import numpy as np
import streamlit as st
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Any

# ---------------------------
# Safety: cv2 import guard
# ---------------------------
try:
    import cv2  # explicit for Cloud build
except Exception as e:
    st.error(
        "OpenCV import failed. On Streamlit Cloud:\n"
        "• Add 'libgl1', 'libglib2.0-0', 'libstdc++6' to packages.txt\n"
        "• Use 'opencv-python-headless' in requirements.txt\n\n"
        f"Error:\n{e}"
    )
    st.stop()

# ---------------------------
# Utils
# ---------------------------
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def save_image(path: Path, image_bgr: np.ndarray, quality: int = 95):
    ensure_dir(path.parent)
    cv2.imwrite(str(path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])

def zip_directory(src_dir: Path, zip_path: Path):
    ensure_dir(zip_path.parent)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for f in files:
                fp = Path(root) / f
                zf.write(fp, fp.relative_to(src_dir))

def make_zip_in_memory(file_tuples: List[Tuple[str, bytes]]) -> io.BytesIO:
    """
    Build a ZIP (in memory). file_tuples: List of (zip_path_inside, bytes)
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for zpath, data in file_tuples:
            zf.writestr(zpath, data)
    buf.seek(0)
    return buf

# ---------------------------
# EL Preprocessing (OpenCV)
# ---------------------------
def normalize_el(img_bgr: np.ndarray, clip_limit_frac: float = 0.02, tile_size: int = 8, sigma: float = 0.6) -> np.ndarray:
    """
    CLAHE + optional Gaussian blur. Returns uint8 grayscale.
    clip_limit_frac: fractional slider mapped to OpenCV clipLimit ~ [1..10].
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cv_clip = max(1.0, min(10.0, clip_limit_frac * 100.0))
    clahe = cv2.createCLAHE(clipLimit=cv_clip, tileGridSize=(tile_size, tile_size))
    gray_norm = clahe.apply(gray)
    if sigma and sigma > 0:
        k = max(3, int(2 * sigma) * 2 + 1)  # odd kernel
        gray_norm = cv2.GaussianBlur(gray_norm, (k, k), 0)
    return gray_norm

# ---------------------------
# Alignment (optional)
# ---------------------------
def perspective_warp(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) != 4:
        return img_bgr
    pts = approx.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    rect = np.array([tl, tr, br, bl], dtype=np.float32)
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxW = int(max(widthA, widthB))
    maxH = int(max(heightA, heightB))
    if maxW < 100 or maxH < 100:
        return img_bgr
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img_bgr, M, (maxW, maxH), flags=cv2.INTER_LINEAR)

def auto_deskew(img_bgr: np.ndarray, gray_u8: np.ndarray, hough_thresh: int = 120) -> np.ndarray:
    edges = cv2.Canny(gray_u8, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, hough_thresh)
    if lines is None:
        return img_bgr
    angles = []
    for l in lines[:200]:
        theta = l[0][1]
        deg = np.rad2deg(theta)
        deg = ((deg + 90) % 180) - 90
        angles.append(deg)
    if not angles:
        return img_bgr
    mean_angle = float(np.mean(angles))
    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), -mean_angle, 1.0)
    return cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

# ---------------------------
# Grid detection & cell building (OpenCV)
# ---------------------------
def binarize(gray_u8: np.ndarray, mode: str = "otsu") -> np.ndarray:
    if mode == "adaptive":
        return cv2.adaptiveThreshold(gray_u8, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 5)
    _, bw = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return bw

def detect_grid_lines(gray_u8: np.ndarray,
                      polarity: str = "auto",
                      binarize_mode: str = "otsu",
                      ksize_v: int = 25,
                      ksize_h: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    bw = binarize(gray_u8, mode=binarize_mode)
    if polarity == "auto":
        use = 255 - bw if np.mean(gray_u8) > 127 else bw
    elif polarity == "dark":
        use = 255 - bw
    else:
        use = bw
    use_u8 = (use > 0).astype(np.uint8) * 255
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ksize_v))
    vert = cv2.dilate(cv2.erode(use_u8, kernel_v, iterations=1), kernel_v, iterations=1)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize_h, 1))
    horiz = cv2.dilate(cv2.erode(use_u8, kernel_h, iterations=1), kernel_h, iterations=1)
    return vert, horiz

def project_peaks(line_map_u8: np.ndarray, axis: int = 0, min_dist: int = 30, min_strength: float = 0.3) -> List[int]:
    proj = line_map_u8.sum(axis=axis).astype(np.float64)
    rng = proj.max() - proj.min()
    proj_norm = (proj - proj.min()) / (rng + 1e-6)
    peaks, last = [], -min_dist
    for i in range(1, len(proj_norm) - 1):
        if proj_norm[i] > min_strength and proj_norm[i] > proj_norm[i - 1] and proj_norm[i] > proj_norm[i + 1]:
            if i - last >= min_dist:
                peaks.append(i); last = i
    return peaks

def cuts_from_peaks(peaks: List[int], maxlen: int) -> List[int]:
    if len(peaks) < 2:
        return [0, maxlen - 1]
    cuts = [0] + [ (peaks[i] + peaks[i+1]) // 2 for i in range(len(peaks)-1) ] + [maxlen - 1]
    return sorted(list(set(cuts)))

def build_cells_from_grid(img_bgr: np.ndarray,
                          vert_map_u8: np.ndarray,
                          horiz_map_u8: np.ndarray,
                          min_cell_w: int = 40,
                          min_cell_h: int = 40) -> List[Dict[str, Any]]:
    H, W = vert_map_u8.shape
    xs = project_peaks(vert_map_u8, axis=0, min_dist=max(20, W // 30))
    ys = project_peaks(horiz_map_u8, axis=1, min_dist=max(20, H // 30))
    xcuts = cuts_from_peaks(xs, W)
    ycuts = cuts_from_peaks(ys, H)
    cells = []
    for r in range(len(ycuts)-1):
        y0, y1 = ycuts[r], ycuts[r+1]
        for c in range(len(xcuts)-1):
            x0, x1 = xcuts[c], xcuts[c+1]
            w, h = x1 - x0, y1 - y0
            if w >= min_cell_w and h >= min_cell_h:
                crop = img_bgr[y0:y1, x0:x1].copy()
                cells.append({"row": r, "col": c, "bbox": (x0, y0, x1, y1), "image": crop})
    return cells

def overlay_grid(img_bgr: np.ndarray, cells: List[Dict[str, Any]], color=(0, 255, 0), thickness=2) -> np.ndarray:
    vis = img_bgr.copy()
    for cell in cells:
        x0, y0, x1, y1 = cell["bbox"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, thickness)
    return vis

def manual_split(img_bgr: np.ndarray, n_rows: int, n_cols: int, margin: int = 0) -> List[Dict[str, Any]]:
    h, w = img_bgr.shape[:2]
    x0, y0 = margin, margin
    x1, y1 = w - margin, h - margin
    cell_w = (x1 - x0) // n_cols
    cell_h = (y1 - y0) // n_rows
    cells = []
    for r in range(n_rows):
        for c in range(n_cols):
            cx0 = x0 + c * cell_w
            cy0 = y0 + r * cell_h
            cx1 = cx0 + cell_w
            cy1 = cy0 + cell_h
            crop = img_bgr[cy0:cy1, cx0:cx1].copy()
            cells.append({"row": r, "col": c, "bbox": (cx0, cy0, cx1, cy1), "image": crop})
    return cells

# ---------------------------
# YOLOv5 (ONNX) via OpenCV DNN
# ---------------------------
def letterbox(im: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize with unchanged aspect ratio using padding (like YOLO). Return resized image, ratio, dwdh.
    """
    shape = im.shape[:2]  # H, W
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    # x: [n,4] with cx,cy,w,h
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y

def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float = 0.45) -> List[int]:
    """
    Simple IoU-based NMS for xyxy boxes.
    """
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        rest = idxs[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_rest - inter + 1e-6)
        idxs = rest[iou < iou_thres]
    return keep

def run_yolov5_onnx(net: cv2.dnn_Net,
                    img_bgr: np.ndarray,
                    input_size: int = 640,
                    conf_thres: float = 0.25,
                    iou_thres: float = 0.45) -> Dict[str, Any]:
    """
    Run YOLOv5 ONNX via OpenCV DNN on a BGR cell image.
    Assumes ONNX exported with decoded heads (end-to-end), producing predictions of shape [N, 5+C] per image.
    Returns dict: {'boxes': xyxy int list, 'scores': list, 'classes': list}
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized, r, dwdh = letterbox(img_rgb, (input_size, input_size))
    blob = resized.transpose(2, 0, 1)  # HWC -> CHW
    blob = blob[np.newaxis, :, :, :].astype(np.float32) / 255.0

    net.setInput(blob)
    pred = net.forward()  # expected [1, N, 5+C] or [N, 5+C]
    pred = np.squeeze(pred)  # [N, 5+C]
    if pred.ndim != 2 or pred.shape[1] < 6:
        # Model format not compatible
        return {"boxes": [], "scores": [], "classes": []}

    # Parse: [cx, cy, w, h, obj_conf, cls_scores...]
    boxes = pred[:, :4]
    obj_conf = pred[:, 4]
    cls_scores = pred[:, 5:]
    class_ids = cls_scores.argmax(axis=1)
    class_conf = cls_scores[np.arange(cls_scores.shape[0]), class_ids]
    conf = obj_conf * class_conf

    keep_mask = conf >= conf_thres
    boxes = boxes[keep_mask]
    conf = conf[keep_mask]
    class_ids = class_ids[keep_mask]

    if boxes.size == 0:
        return {"boxes": [], "scores": [], "classes": []}

    # Convert to xyxy in original cell scale
    boxes = xywh2xyxy(boxes)
    # remove letterbox padding and scale back
    boxes[:, [0, 2]] -= dwdh[0]
    boxes[:, [1, 3]] -= dwdh[1]
    boxes /= r

    H, W = img_rgb.shape[:2]
    boxes[:, 0] = np.clip(boxes[:, 0], 0, W - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, H - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, W - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, H - 1)

    # NMS
    keep_idx = nms_xyxy(boxes, conf, iou_thres)
    boxes = boxes[keep_idx].astype(int)
    conf = conf[keep_idx]
    class_ids = class_ids[keep_idx]

    return {
        "boxes": [b.tolist() for b in boxes],
        "scores": [float(s) for s in conf],
        "classes": [int(c) for c in class_ids]
    }

def color_for_class(cls_id: int) -> Tuple[int, int, int]:
    np.random.seed(cls_id + 19)
    c = np.random.randint(0, 255, size=3)
    return int(c[0]), int(c[1]), int(c[2])

def overlay_detections(img_bgr: np.ndarray,
                       det: Dict[str, Any],
                       class_names: List[str]) -> np.ndarray:
    vis = img_bgr.copy()
    for box, cls, score in zip(det["boxes"], det["classes"], det["scores"]):
        x1, y1, x2, y2 = box
        color = color_for_class(cls)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[cls] if cls < len(class_names) else cls}:{score:.2f}"
        cv2.putText(vis, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="EL Cells + YOLOv5 Defect Detection (OpenCV-only)", layout="wide")
st.title("🔬 EL PV Module → Cell Segregation + YOLOv5 Defect Detection (OpenCV)")

st.markdown("""
1) Upload EL module images → auto-segment into cells (OpenCV-only).
2) Upload YOLOv5 **ONNX** model to detect defects **per cell**.
3) Save only **defective cells** (crops + overlays) and download as ZIP.
""")

# Sidebar: EL segmentation
st.sidebar.header("⚙️ EL Segmentation")
clip_limit = st.sidebar.slider("CLAHE clipLimit (fraction → mapped)", 0.005, 0.05, 0.02, 0.005)
tile_size  = st.sidebar.slider("CLAHE tile size", 4, 32, 8, 2)
sigma      = st.sidebar.slider("Gaussian sigma", 0.0, 2.0, 0.6, 0.1)
do_warp    = st.sidebar.checkbox("Perspective warp", True)
do_deskew  = st.sidebar.checkbox("Auto deskew", True)
polarity   = st.sidebar.selectbox("Line polarity", ["auto", "dark", "bright"], index=0)
binarize_mode = st.sidebar.selectbox("Binarization", ["otsu", "adaptive"], index=0)
ksize_v    = st.sidebar.slider("Vertical kernel size", 5, 75, 25, 1)
ksize_h    = st.sidebar.slider("Horizontal kernel size", 5, 75, 25, 1)
min_cell_w = st.sidebar.slider("Min cell width (px)", 20, 400, 40, 10)
min_cell_h = st.sidebar.slider("Min cell height (px)", 20, 400, 40, 10)

# Manual layout
st.sidebar.header("📐 Layout preset (optional)")
layout_preset = st.sidebar.selectbox("Preset", ["None", "6×24 (144)", "12×12 (144)"], index=0)
use_manual    = st.sidebar.checkbox("Use manual rows×cols", False)
n_rows        = st.sidebar.number_input("Rows", 1, 30, 6)
n_cols        = st.sidebar.number_input("Cols", 1, 48, 10)
manual_margin = st.sidebar.number_input("Manual margin (px)", 0, 200, 0)

if layout_preset == "6×24 (144)":
    use_manual = True; n_rows, n_cols = 6, 24
elif layout_preset == "12×12 (144)":
    use_manual = True; n_rows, n_cols = 12, 12

# Sidebar: YOLOv5 detection
st.sidebar.header("🤖 YOLOv5 Defect Detection")
yolo_file = st.sidebar.file_uploader("Upload YOLOv5 ONNX model (.onnx)", type=["onnx"])
input_size = st.sidebar.selectbox("Model input size", [640, 512, 416], index=0)
conf_thres = st.sidebar.slider("Confidence (conf)", 0.05, 0.95, 0.25, 0.05)
iou_thres  = st.sidebar.slider("IoU (NMS)", 0.10, 0.95, 0.45, 0.05)
class_names_str = st.sidebar.text_area("Class names (comma-separated)", "crack, hotspot, snail_trail, inactive_area")
class_names = [s.strip() for s in class_names_str.split(",") if s.strip()]

# Output
st.sidebar.header("💾 Output")
out_dir_str = st.sidebar.text_input("Output directory", "output")
save_defect_zip_to_disk = st.sidebar.checkbox("Also save defects ZIP to disk", True)

start_btn = st.sidebar.button("🚀 Run")

uploads = st.file_uploader("Upload EL module image(s)", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"], accept_multiple_files=True)

# ---------------------------
# Main run
# ---------------------------
if start_btn:
    out_base = Path(out_dir_str); ensure_dir(out_base)

    # Load YOLOv5 ONNX (optional but recommended)
    net = None
    onnx_path = None
    if yolo_file is not None:
        onnx_path = Path("models") / yolo_file.name
        ensure_dir(onnx_path.parent)
        with open(onnx_path, "wb") as f:
            f.write(yolo_file.read())
        try:
            net = cv2.dnn.readNetFromONNX(str(onnx_path))
        except Exception as e:
            st.error(f"Failed to load ONNX model: {e}")
            net = None

    if not uploads:
        st.warning("Please upload at least one EL module image.")
    else:
        for upl in uploads:
            img_pil = Image.open(io.BytesIO(upl.read())).convert("RGB")
            img_bgr = pil_to_cv(img_pil)
            save_dir = out_base / Path(upl.name).stem
            ensure_dir(save_dir)

            # Optional alignment
            if do_warp:
                img_bgr = perspective_warp(img_bgr)
            gray_u8 = normalize_el(img_bgr, clip_limit_frac=clip_limit, tile_size=tile_size, sigma=sigma)
            if do_deskew:
                img_bgr = auto_deskew(img_bgr, gray_u8)
                gray_u8 = normalize_el(img_bgr, clip_limit_frac=clip_limit, tile_size=tile_size, sigma=sigma)

            # Split into cells
            if not use_manual:
                vert_map, horiz_map = detect_grid_lines(gray_u8, polarity=polarity, binarize_mode=binarize_mode, ksize_v=ksize_v, ksize_h=ksize_h)
                cells = build_cells_from_grid(img_bgr, vert_map, horiz_map, min_cell_w=min_cell_w, min_cell_h=min_cell_h)
            else:
                cells = manual_split(img_bgr, n_rows=int(n_rows), n_cols=int(n_cols), margin=int(manual_margin))

            grid_overlay = overlay_grid(img_bgr, cells)
            save_image(save_dir / "module_grid.jpg", grid_overlay)

            # Save all cells (optional but useful)
            cells_dir = save_dir / "cells"; ensure_dir(cells_dir)
            for cell in cells:
                r, c = cell["row"], cell["col"]
                save_image(cells_dir / f"cell_{r:02d}_{c:02d}.jpg", cell["image"])

            # YOLOv5 detection per cell (if model provided)
            defects_dir = save_dir / "defects_overlay"; ensure_dir(defects_dir)
            defective_cells_dir = save_dir / "defective_cells"; ensure_dir(defective_cells_dir)
            summary = []
            zip_mem_files = []  # (zip_path_inside, bytes)

            if net is not None:
                for cell in cells:
                    r, c = cell["row"], cell["col"]
                    crop_bgr = cell["image"]
                    det = run_yolov5_onnx(net, crop_bgr, input_size=input_size, conf_thres=conf_thres, iou_thres=iou_thres)

                    num_det = len(det["boxes"])
                    summary.append({
                        "row": r, "col": c, "bbox": cell["bbox"],
                        "num_detections": num_det,
                        "classes": det["classes"],
                        "scores": det["scores"]
                    })

                    if num_det > 0:
                        # Overlay and save
                        overlay = overlay_detections(crop_bgr, det, class_names)
                        save_image(defects_dir / f"cell_{r:02d}_{c:02d}_overlay.jpg", overlay)
                        save_image(defective_cells_dir / f"cell_{r:02d}_{c:02d}.jpg", crop_bgr)

                        # Add both images to in-memory ZIP list
                        ok1, enc1 = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        ok2, enc2 = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        if ok1:
                            zip_mem_files.append((f"defects_overlay/cell_{r:02d}_{c:02d}_overlay.jpg", enc1.tobytes()))
                        if ok2:
                            zip_mem_files.append((f"defective_cells/cell_{r:02d}_{c:02d}.jpg", enc2.tobytes()))

                # Save summary JSON
                with open(save_dir / "defects_summary.json", "w") as f:
                    json.dump({"n_cells": len(cells), "results": summary}, f, indent=2)

                # In-memory ZIP for defective cells/overlays
                if zip_mem_files:
                    defects_zip_mem = make_zip_in_memory(zip_mem_files)
                    st.download_button(
                        "📦 Download defective cells + overlays (ZIP)",
                        data=defects_zip_mem,
                        file_name=f"{Path(upl.name).stem}_defects.zip",
                        mime="application/zip"
                    )
                    if save_defect_zip_to_disk:
                        disk_zip_path = save_dir / f"{Path(upl.name).stem}_defects.zip"
                        with open(disk_zip_path, "wb") as f:
                            f.write(defects_zip_mem.getvalue())
                        st.info(f"Saved defects ZIP to disk: {disk_zip_path}")
            else:
                st.warning("No YOLOv5 ONNX model uploaded. Skipping defect detection.")

            # Show grid overlay & a preview
            st.success(f"Processed {upl.name}: {len(cells)} cells")
            st.image(cv_to_pil(grid_overlay), caption=f"Cell grid overlay: {upl.name}", use_column_width=True)
            cols_show = st.columns(6)
            for i, cell in enumerate(cells[:12]):
                r, c = cell["row"], cell["col"]
                cols_show[i % len(cols_show)].image(cv_to_pil(cell["image"]), caption=f"Cell r{r} c{c}", use_column_width=True)

            # Full results ZIP (all outputs saved on disk)
            full_zip_path = save_dir / f"{Path(upl.name).stem}_results.zip"
            zip_directory(save_dir, full_zip_path)
            with open(full_zip_path, "rb") as f:
                st.download_button(f"📦 Download all outputs for {upl.name}", data=f, file_name=full_zip_path.name)

st.markdown("---")
st.caption("Tips: Increase vertical/horizontal kernel sizes if gridlines are faint; use 'adaptive' threshold for uneven EL; use 6×24 or 12×12 preset to guarantee 144 cells.")
