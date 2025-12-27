
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
# Safe import guard (Cloud helpful error)
# ---------------------------
try:
    import cv2  # already imported above; repeated here for clarity
except Exception as e:
    st.error(
        "Failed to import OpenCV (cv2). On Streamlit Cloud, ensure:\n"
        "1) packages.txt includes 'libgl1' and 'libglib2.0-0'\n"
        "2) requirements.txt uses 'opencv-python-headless'\n\n"
        f"Error:\n{e}"
    )
    st.stop()

# ---------------------------
# Utilities
# ---------------------------
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

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

# ---------------------------
# EL-specific preprocessing & optional alignment
# ---------------------------
def normalize_el(img_bgr: np.ndarray, clahe_clip: float = 2.5, tile: int = 8, blur_ksize: int = 3) -> np.ndarray:
    """
    CLAHE + optional blur to enhance busbars/gridlines in EL images.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(tile, tile))
    gray_norm = clahe.apply(gray)
    if blur_ksize > 0:
        gray_norm = cv2.GaussianBlur(gray_norm, (blur_ksize, blur_ksize), 0)
    return gray_norm

def perspective_warp(img_bgr: np.ndarray) -> np.ndarray:
    """
    Try to detect the module rectangle and warp to a flat view.
    Useful if your frame is visible and the module is skewed.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
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
    warped = cv2.warpPerspective(img_bgr, M, (maxW, maxH), flags=cv2.INTER_LINEAR)
    return warped

# ---------------------------
# Grid detection + cell building
# ---------------------------
def detect_grid_lines(gray: np.ndarray,
                      polarity: str = "auto",
                      binarize: str = "otsu",
                      ksize_v: int = 25,
                      ksize_h: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect vertical and horizontal line maps via morphology.
    """
    if binarize == "adaptive":
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 31, 5)
    else:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # EL: cells bright, lines often dark → invert BW to focus lines
    if polarity == "auto":
        use = 255 - bw if np.mean(gray) > 127 else bw
    elif polarity == "dark":
        use = 255 - bw
    else:
        use = bw

    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ksize_v))
    vert = cv2.erode(use, kernel_v, iterations=1)
    vert = cv2.dilate(vert, kernel_v, iterations=1)

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize_h, 1))
    horiz = cv2.erode(use, kernel_h, iterations=1)
    horiz = cv2.dilate(horiz, kernel_h, iterations=1)

    return vert, horiz

def project_peaks(line_map: np.ndarray, axis: int = 0, min_dist: int = 30, min_strength: float = 0.3) -> List[int]:
    """
    Find peaks along projections of the line map.
    axis=0 → vertical lines (column projection); axis=1 → horizontal (row projection).
    """
    proj = line_map.sum(axis=axis)
    proj_norm = (proj - proj.min()) / (proj.max() - proj.min() + 1e-6)
    peaks = []
    last_idx = -min_dist
    for i in range(1, len(proj_norm) - 1):
        if proj_norm[i] > min_strength and proj_norm[i] > proj_norm[i - 1] and proj_norm[i] > proj_norm[i + 1]:
            if i - last_idx >= min_dist:
                peaks.append(i)
                last_idx = i
    return peaks

def cuts_from_peaks(peaks: List[int], maxlen: int) -> List[int]:
    """
    Convert peak positions (grid lines) to cut boundaries between cells.
    """
    if len(peaks) < 2:
        return [0, maxlen - 1]
    cuts = [0]
    for i in range(len(peaks) - 1):
        cuts.append((peaks[i] + peaks[i + 1]) // 2)
    cuts.append(maxlen - 1)
    cuts = sorted(list(set(cuts)))
    return cuts

def build_cells_from_grid(img_bgr: np.ndarray,
                          vert_map: np.ndarray,
                          horiz_map: np.ndarray,
                          min_cell_w: int = 40,
                          min_cell_h: int = 40) -> List[Dict[str, Any]]:
    """
    Build cell crops using detected vertical/horizontal splits.
    """
    H, W = vert_map.shape
    xs = project_peaks(vert_map, axis=0, min_dist=max(20, W // 30))
    ys = project_peaks(horiz_map, axis=1, min_dist=max(20, H // 30))

    xcuts = cuts_from_peaks(xs, W)
    ycuts = cuts_from_peaks(ys, H)

    cells = []
    for r in range(len(ycuts) - 1):
        y0, y1 = ycuts[r], ycuts[r + 1]
        for c in range(len(xcuts) - 1):
            x0, x1 = xcuts[c], xcuts[c + 1]
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
    """
    Fallback: evenly split image into n_rows × n_cols cells if auto detection struggles.
    """
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
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="PV EL Module → Cell Segregation", layout="wide")
st.title("🔬 PV EL Module → Cell Segregation (Streamlit)")

st.markdown("""
Upload **EL PV module** images to automatically detect the **cell grid** and export **per‑cell crops**.
If auto detection struggles (partial modules/atypical layouts), use the **Manual rows × cols** fallback.
""")

# Sidebar controls
st.sidebar.header("⚙️ Settings")

# Preprocessing
clahe_clip = st.sidebar.slider("CLAHE clipLimit", 1.0, 4.0, 2.5, 0.1)
clahe_tile = st.sidebar.slider("CLAHE tile size", 4, 16, 8, 1)
blur_ksize = st.sidebar.slider("Gaussian blur (ksize)", 0, 7, 3, 1)

# Alignment
do_warp = st.sidebar.checkbox("Perspective warp (rectify module)", True)

# Detection params
polarity = st.sidebar.selectbox("Line polarity", ["auto", "dark", "bright"], index=0)
binarize = st.sidebar.selectbox("Binarization", ["otsu", "adaptive"], index=0)
ksize_v = st.sidebar.slider("Vertical kernel size", 5, 75, 25, 1)
ksize_h = st.sidebar.slider("Horizontal kernel size", 5, 75, 25, 1)
min_cell_w = st.sidebar.slider("Min cell width (px)", 20, 400, 40, 10)
min_cell_h = st.sidebar.slider("Min cell height (px)", 20, 400, 40, 10)

# Manual fallback
use_manual = st.sidebar.checkbox("Use manual rows × cols fallback", False)
n_rows = st.sidebar.number_input("Rows", min_value=1, max_value=20, value=6)
n_cols = st.sidebar.number_input("Cols", min_value=1, max_value=24, value=10)
manual_margin = st.sidebar.number_input("Manual margin (px)", min_value=0, max_value=200, value=0)

# Output options
out_dir_str = st.sidebar.text_input("Output directory", "output")
start_btn = st.sidebar.button("🚀 Run")

# Uploader
uploads = st.file_uploader("Upload EL module image(s)", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"], accept_multiple_files=True)

# ---------------------------
# Processing
# ---------------------------
def process_single_image(img_pil: Image.Image,
                         settings: Dict[str, Any],
                         save_root: Path) -> Dict[str, Any]:
    t0 = time.time()
    img_bgr = pil_to_cv(img_pil)

    # Optional perspective warp first
    if settings["do_warp"]:
        img_bgr = perspective_warp(img_bgr)

    # EL preprocessing
    gray_norm = normalize_el(img_bgr,
                             clahe_clip=settings["clahe_clip"],
                             tile=settings["clahe_tile"],
                             blur_ksize=settings["blur_ksize"])

    # Auto detection or manual split
    if not settings["use_manual"]:
        vert_map, horiz_map = detect_grid_lines(gray_norm,
                                                polarity=settings["polarity"],
                                                binarize=settings["binarize"],
                                                ksize_v=settings["ksize_v"],
                                                ksize_h=settings["ksize_h"])
        cells = build_cells_from_grid(img_bgr, vert_map, horiz_map,
                                      min_cell_w=settings["min_cell_w"],
                                      min_cell_h=settings["min_cell_h"])
    else:
        cells = manual_split(img_bgr, n_rows=settings["n_rows"], n_cols=settings["n_cols"], margin=settings["manual_margin"])

    grid_overlay = overlay_grid(img_bgr, cells)

    # Save outputs
    ensure_dir(save_root)
    save_image(save_root / "module_grid.jpg", grid_overlay)
    cells_dir = save_root / "cells"
    ensure_dir(cells_dir)

    meta = []
    for cell in cells:
        r, c = cell["row"], cell["col"]
        save_image(cells_dir / f"cell_{r:02d}_{c:02d}.jpg", cell["image"])
        meta.append({"row": r, "col": c, "bbox": cell["bbox"]})

    with open(save_root / "summary.json", "w") as f:
        json.dump({"n_cells": len(cells), "cells": meta}, f, indent=2)

    return {
        "n_cells": len(cells),
        "grid_overlay": grid_overlay,
        "cells": cells,
        "elapsed": time.time() - t0
    }

# Run pipeline
if start_btn:
    out_base = Path(out_dir_str)
    ensure_dir(out_base)

    if not uploads:
        st.warning("Please upload at least one image.")
    else:
        for upl in uploads:
            img_pil = Image.open(io.BytesIO(upl.read())).convert("RGB")
            save_dir = out_base / Path(upl.name).stem

            res = process_single_image(
                img_pil,
                settings={
                    "clahe_clip": clahe_clip,
                    "clahe_tile": clahe_tile,
                    "blur_ksize": blur_ksize,
                    "do_warp": do_warp,
                    "polarity": polarity,
                    "binarize": binarize,
                    "ksize_v": ksize_v,
                    "ksize_h": ksize_h,
                    "min_cell_w": min_cell_w,
                    "min_cell_h": min_cell_h,
                    "use_manual": use_manual,
                    "n_rows": int(n_rows),
                    "n_cols": int(n_cols),
                    "manual_margin": int(manual_margin)
                },
                save_root=save_dir
            )

            st.success(f"Processed {upl.name}: {res['n_cells']} cells in {res['elapsed']:.2f}s")
            st.image(cv_to_pil(res["grid_overlay"]), caption=f"Detected cell grid: {upl.name}", use_column_width=True)

            # Preview: first 12 cells
            cols_show = st.columns(min(6, max(1, int(n_cols))))
            preview_count = min(len(res["cells"]), 12)
            for i in range(preview_count):
                r, c = res["cells"][i]["row"], res["cells"][i]["col"]
                cols_show[i % len(cols_show)].image(
                    cv_to_pil(res["cells"][i]["image"]),
                    caption=f"Cell r{r} c{c}",
                    use_column_width=True
                )

            # ZIP download for this image
            zip_path = save_dir / f"{Path(upl.name).stem}_cells.zip"
            zip_directory(save_dir, zip_path)
            with open(zip_path, "rb") as f:
                st.download_button(f"📦 Download crops for {upl.name}", data=f, file_name=zip_path.name)

st.markdown("---")
st.caption("Tips: EL images typically have darker busbars/gridlines than cells. Increase kernel sizes if lines are faint; use 'adaptive' threshold for uneven brightness; use manual rows×cols for standard layouts (e.g., 6×10).")
