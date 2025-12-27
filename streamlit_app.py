import os
import io
import cv2
import json
import time
import zipfile
import numpy as np
import streamlit as st
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Optional: only import ultralytics when running inference to avoid slow startup
from ultralytics import YOLO

# ---------------------------
# Utility Functions
# ---------------------------

def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def save_image(path: Path, image: np.ndarray, quality: int = 95):
    ensure_dir(path.parent)
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])

def zip_directory(src_dir: Path, zip_path: Path):
    ensure_dir(zip_path.parent)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for f in files:
                fp = Path(root) / f
                zf.write(fp, fp.relative_to(src_dir))

def normalize_image_for_grid(img_bgr: np.ndarray) -> np.ndarray:
    """
    Normalize brightness/contrast to enhance busbars/grid-lines. Works for EL/IR/visible images.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # CLAHE to normalize contrast
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_norm = clahe.apply(gray)
    # Slight Gaussian blur to reduce speckle
    gray_blur = cv2.GaussianBlur(gray_norm, (3, 3), 0)
    return gray_blur

def detect_grid_lines(gray: np.ndarray,
                      ksize_v: int = 25,
                      ksize_h: int = 25,
                      thresh: int = 150) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect vertical and horizontal line maps using morphological operations.
    """
    # Binary threshold (adaptive can be used too)
    _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Invert if lines are dark
    # Decide polarity based on mean
    if np.mean(gray) > 127:  # bright background
        bw_inv = 255 - bw
    else:
        bw_inv = bw

    # Vertical lines
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ksize_v))
    vert = cv2.erode(bw_inv, kernel_v, iterations=1)
    vert = cv2.dilate(vert, kernel_v, iterations=1)

    # Horizontal lines
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize_h, 1))
    horiz = cv2.erode(bw_inv, kernel_h, iterations=1)
    horiz = cv2.dilate(horiz, kernel_h, iterations=1)

    return vert, horiz

def project_peaks(line_map: np.ndarray, axis: int = 0, min_dist: int = 30, min_strength: float = 0.3) -> List[int]:
    """
    Find peaks along projections of line_map (sum over rows/cols).
    axis=0 for vertical lines (sum over rows → column projection),
    axis=1 for horizontal lines (sum over cols → row projection).
    """
    proj = line_map.sum(axis=axis)
    proj_norm = (proj - proj.min()) / (proj.max() - proj.min() + 1e-6)
    # Simple peak picking: local maxima with min distance
    peaks = []
    last_idx = -min_dist
    for i in range(1, len(proj_norm) - 1):
        if proj_norm[i] > min_strength and proj_norm[i] > proj_norm[i - 1] and proj_norm[i] > proj_norm[i + 1]:
            if i - last_idx >= min_dist:
                peaks.append(i)
                last_idx = i
    return peaks

def build_cells_from_grid(img_bgr: np.ndarray,
                          vert_map: np.ndarray,
                          horiz_map: np.ndarray,
                          min_cell_w: int = 40,
                          min_cell_h: int = 40) -> List[Dict[str, Any]]:
    """
    Construct cell bounding boxes from vertical/horizontal splits.
    Returns list of dicts with bbox and cropped images.
    """
    H, W = vert_map.shape
    xs = project_peaks(vert_map, axis=0, min_dist=max(20, W // 30))
    ys = project_peaks(horiz_map, axis=1, min_dist=max(20, H // 30))

    # Convert peak positions to cuts (intermediate between lines)
    def to_cuts(peaks: List[int], maxlen: int) -> List[int]:
        cuts = [0]
        for i in range(len(peaks) - 1):
            cuts.append((peaks[i] + peaks[i + 1]) // 2)
        cuts.append(maxlen - 1)
        # Remove duplicates, ensure sorted
        cuts = sorted(list(set(cuts)))
        return cuts

    xcuts = to_cuts(xs, W)
    ycuts = to_cuts(ys, H)

    cells = []
    for r in range(len(ycuts) - 1):
        y0, y1 = ycuts[r], ycuts[r + 1]
        for c in range(len(xcuts) - 1):
            x0, x1 = xcuts[c], xcuts[c + 1]
            w, h = x1 - x0, y1 - y0
            if w >= min_cell_w and h >= min_cell_h:
                crop = img_bgr[y0:y1, x0:x1].copy()
                cells.append({
                    "row": r,
                    "col": c,
                    "bbox": (x0, y0, x1, y1),
                    "image": crop
                })
    return cells

def overlay_grid(img_bgr: np.ndarray, cells: List[Dict[str, Any]], color=(0, 255, 0), thickness=2) -> np.ndarray:
    vis = img_bgr.copy()
    for cell in cells:
        x0, y0, x1, y1 = cell["bbox"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, thickness)
    return vis

def run_yolo_on_image(model: YOLO,
                      img_bgr: np.ndarray,
                      conf: float = 0.25,
                      iou: float = 0.45,
                      task_type: str = "detect") -> Dict[str, Any]:
    """
    Run YOLO on an image and return parsed results.
    Supports 'detect' (boxes) and 'segment' (masks).
    """
    # Ultralytics expects RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = model.predict(source=img_rgb, conf=conf, iou=iou, verbose=False)
    out = {"boxes": [], "masks": [], "classes": [], "scores": []}
    if not results:
        return out

    r = results[0]
    # Boxes
    if r.boxes is not None:
        for b in r.boxes:
            xyxy = b.xyxy.cpu().numpy().astype(int)[0]  # [x1,y1,x2,y2]
            cls_id = int(b.cls.cpu().numpy()[0])
            score = float(b.conf.cpu().numpy()[0])
            out["boxes"].append(xyxy.tolist())
            out["classes"].append(cls_id)
            out["scores"].append(score)

    # Masks (seg models)
    if task_type == "segment" and r.masks is not None and r.masks.data is not None:
        # r.masks.data is [N, H, W] normalized
        masks = r.masks.data.cpu().numpy()
        for m in masks:
            out["masks"].append((m > 0.5).astype(np.uint8))  # binary

    return out

def color_for_class(cls_id: int) -> Tuple[int, int, int]:
    # Simple deterministic color palette
    np.random.seed(cls_id + 7)
    c = np.random.randint(0, 255, size=3).tolist()
    return (int(c[0]), int(c[1]), int(c[2]))

def overlay_detections(img_bgr: np.ndarray,
                       detections: Dict[str, Any],
                       class_names: List[str],
                       task_type: str = "detect",
                       alpha: float = 0.4) -> np.ndarray:
    """
    Draw boxes or masks + labels on image.
    """
    vis = img_bgr.copy()

    if task_type == "detect":
        for box, cls_id, score in zip(detections["boxes"], detections["classes"], detections["scores"]):
            x1, y1, x2, y2 = box
            color = color_for_class(cls_id)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{class_names[cls_id] if cls_id < len(class_names) else cls_id}: {score:.2f}"
            cv2.putText(vis, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    elif task_type == "segment":
        overlay = np.zeros_like(vis)
        for idx, (mask, cls_id, score) in enumerate(zip(detections["masks"], detections["classes"], detections["scores"])):
            color = np.array(color_for_class(cls_id), dtype=np.uint8)
            # Broadcast color over mask
            colored_mask = np.zeros_like(vis)
            for ch in range(3):
                colored_mask[:, :, ch] = mask * color[ch]
            overlay = cv2.add(overlay, colored_mask)
            # draw a label at centroid
            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                x_c, y_c = int(xs.mean()), int(ys.mean())
                label = f"{class_names[cls_id] if cls_id < len(class_names) else cls_id}: {score:.2f}"
                cv2.putText(vis, label, (x_c, y_c), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        vis = cv2.addWeighted(vis, 1.0, overlay, alpha, 0)

    return vis

def process_module_image(img_pil: Image.Image,
                         model_path: Path,
                         task_type: str,
                         conf: float,
                         iou: float,
                         grid_params: Dict[str, Any],
                         class_names: List[str],
                         save_dir: Path,
                         save_masks: bool = True) -> Dict[str, Any]:
    """
    Full pipeline for one module image: split cells, run YOLO per cell, overlay & save.
    """
    t0 = time.time()
    ensure_dir(save_dir)
    img_bgr = pil_to_cv(img_pil)
    gray = normalize_image_for_grid(img_bgr)
    vert, horiz = detect_grid_lines(
        gray,
        ksize_v=grid_params.get("ksize_v", 25),
        ksize_h=grid_params.get("ksize_h", 25),
        thresh=grid_params.get("thresh", 0)
    )
    cells = build_cells_from_grid(
        img_bgr,
        vert,
        horiz,
        min_cell_w=grid_params.get("min_cell_w", 40),
        min_cell_h=grid_params.get("min_cell_h", 40)
    )

    # Visualization of cell grid
    grid_overlay = overlay_grid(img_bgr, cells)
    save_image(save_dir / "module_grid.jpg", grid_overlay)

    # Load YOLO model
    model = YOLO(str(model_path))

    results_summary = []
    cells_dir = save_dir / "cells"
    ensure_dir(cells_dir)

    defects_dir = save_dir / "defects_overlay"
    ensure_dir(defects_dir)

    masks_dir = save_dir / "masks"
    ensure_dir(masks_dir)

    for idx, cell in enumerate(cells):
        crop_bgr = cell["image"]
        det = run_yolo_on_image(model, crop_bgr, conf=conf, iou=iou, task_type=task_type)
        # Overlay defects on cell
        overlay = overlay_detections(crop_bgr, det, class_names, task_type)
        save_image(cells_dir / f"cell_{cell['row']:02d}_{cell['col']:02d}.jpg", crop_bgr)
        save_image(defects_dir / f"cell_{cell['row']:02d}_{cell['col']:02d}_overlay.jpg", overlay)

        # Save masks if segmentation
        if task_type == "segment" and save_masks and len(det["masks"]) > 0:
            # Save each mask as PNG
            for m_i, mask in enumerate(det["masks"]):
                mask_img = (mask * 255).astype(np.uint8)
                cv2.imwrite(str(masks_dir / f"cell_{cell['row']:02d}_{cell['col']:02d}_mask_{m_i}.png"), mask_img)

        # Collect summary
        results_summary.append({
            "row": cell["row"],
            "col": cell["col"],
            "bbox": cell["bbox"],
            "num_defects": len(det["boxes"]) if task_type == "detect" else len(det["masks"]),
            "classes": det["classes"],
            "scores": det["scores"]
        })

    # Save JSON summary
    with open(save_dir / "summary.json", "w") as f:
        json.dump({
            "n_cells": len(cells),
            "results": results_summary
        }, f, indent=2)

    elapsed = time.time() - t0
    return {
        "cells": cells,
        "grid_overlay": grid_overlay,
        "summary": results_summary,
        "elapsed": elapsed,
        "n_cells": len(cells)
    }

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="PV Module Cell Segmentation & Defect Detection (YOLO)", layout="wide")

st.title("🔍 PV Module → Cell Segmentation + Defect Detection (YOLO)")

st.markdown("""
Upload PV module images (EL/IR/daylight), automatically **segregate PV cells**, and run a **YOLOv8** model to detect or segment defects.
Outputs (cells, overlays, masks, JSON summary) are saved to the selected output folder.
""")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

model_path_str = st.sidebar.text_input("YOLO model path (.pt)", "models/yolov8-pv-defects.pt")
task_type = st.sidebar.selectbox("YOLO task type", ["detect", "segment"], index=0)
conf = st.sidebar.slider("Confidence threshold (conf)", 0.05, 0.95, 0.25, 0.05)
iou = st.sidebar.slider("IoU threshold (NMS)", 0.10, 0.95, 0.45, 0.05)

st.sidebar.subheader("Grid detection parameters")
ksize_v = st.sidebar.slider("Vertical kernel size", 5, 75, 25, 1)
ksize_h = st.sidebar.slider("Horizontal kernel size", 5, 75, 25, 1)
min_cell_w = st.sidebar.slider("Min cell width (px)", 20, 400, 40, 10)
min_cell_h = st.sidebar.slider("Min cell height (px)", 20, 400, 40, 10)
thresh = st.sidebar.slider("Threshold (0=OTSU)", 0, 255, 0, 1)

out_dir_str = st.sidebar.text_input("Output directory", "output")
save_masks = st.sidebar.checkbox("Save segmentation masks (for segment models)", True)

class_names_str = st.sidebar.text_area("Class names (comma-separated)", "crack, hotspot, snail_trail, finger_break, inactive_area")
class_names = [c.strip() for c in class_names_str.split(",") if c.strip()]

st.sidebar.markdown("---")
batch_mode = st.sidebar.checkbox("Batch mode (process all images in a folder)", False)
start_btn = st.sidebar.button("🚀 Run")

# File uploader(s)
if not batch_mode:
    uploads = st.file_uploader("Upload module image(s)", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"], accept_multiple_files=True)
else:
    st.info("Batch mode: All images in the specified input folder will be processed.")
    input_dir_str = st.text_input("Input folder path", "input")

# Main logic
if start_btn:
    out_base = Path(out_dir_str)
    ensure_dir(out_base)
    model_path = Path(model_path_str)

    if not model_path.exists():
        st.error(f"Model not found at: {model_path}. Please provide a valid .pt file.")
    else:
        if batch_mode:
            input_dir = Path(input_dir_str)
            if not input_dir.exists():
                st.error(f"Input dir not found: {input_dir}")
            else:
                images = []
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]:
                    images.extend(list(input_dir.glob(ext)))
                if not images:
                    st.warning("No images found in input folder.")
                else:
                    st.write(f"Found {len(images)} images. Processing...")
                    progress = st.progress(0.0)
                    for i, img_path in enumerate(images):
                        img_pil = Image.open(img_path).convert("RGB")
                        save_dir = out_base / img_path.stem
                        res = process_module_image(
                            img_pil=img_pil,
                            model_path=model_path,
                            task_type=task_type,
                            conf=conf,
                            iou=iou,
                            grid_params={
                                "ksize_v": ksize_v,
                                "ksize_h": ksize_h,
                                "min_cell_w": min_cell_w,
                                "min_cell_h": min_cell_h,
                                "thresh": thresh
                            },
                            class_names=class_names,
                            save_dir=save_dir,
                            save_masks=save_masks
                        )
                        st.write(f"**{img_path.name}** → {res['n_cells']} cells, time: {res['elapsed']:.2f}s")
                        # Show grid overlay
                        st.image(cv_to_pil(res["grid_overlay"]), caption=f"Grid overlay: {img_path.name}", use_column_width=True)
                        progress.progress((i + 1) / len(images))
                    # Zip
                    zip_path = out_base / "batch_results.zip"
                    zip_directory(out_base, zip_path)
                    with open(zip_path, "rb") as f:
                        st.download_button("📦 Download all results (ZIP)", data=f, file_name="batch_results.zip")
        else:
            if not uploads:
                st.warning("Please upload at least one image.")
            else:
                for upl in uploads:
                    img_pil = Image.open(io.BytesIO(upl.read())).convert("RGB")
                    save_dir = out_base / Path(upl.name).stem
                    res = process_module_image(
                        img_pil=img_pil,
                        model_path=model_path,
                        task_type=task_type,
                        conf=conf,
                        iou=iou,
                        grid_params={
                            "ksize_v": ksize_v,
                            "ksize_h": ksize_h,
                            "min_cell_w": min_cell_w,
                            "min_cell_h": min_cell_h,
                            "thresh": thresh
                        },
                        class_names=class_names,
                        save_dir=save_dir,
                        save_masks=save_masks
                    )
                    st.success(f"Processed {upl.name}: {res['n_cells']} cells found in {res['elapsed']:.2f}s")
                    # Show grid overlay
                    st.image(cv_to_pil(res["grid_overlay"]), caption=f"Grid overlay: {upl.name}", use_column_width=True)

                    # Let user download per-image zip
                    zip_path = save_dir / f"{Path(upl.name).stem}_results.zip"
                    zip_directory(save_dir, zip_path)
                    with open(zip_path, "rb") as f:
                        st.download_button(f"📦 Download results for {upl.name}", data=f, file_name=zip_path.name)

st.markdown("---")
st.caption("Tip: Tune kernel sizes and thresholds for different image modalities (EL/IR/visible). Use YOLO segmentation for polygon masks.")
