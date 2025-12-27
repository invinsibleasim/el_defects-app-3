
import os
import io
import time
import json
import zipfile
import numpy as np
import streamlit as st
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Any

# scikit-image imports (no OpenCV)
from skimage import color, exposure, filters, morphology, measure, util
from skimage.filters import threshold_otsu
from skimage.morphology import rectangle

# ---------------------------
# Utilities
# ---------------------------
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def pil_to_np(img: Image.Image) -> np.ndarray:
    """PIL RGB -> NumPy RGB uint8"""
    return np.array(img.convert("RGB"))

def np_to_pil(arr: np.ndarray) -> Image.Image:
    """NumPy RGB uint8 -> PIL Image"""
    return Image.fromarray(arr)

def save_image(path: Path, image_arr: np.ndarray, quality: int = 95):
    ensure_dir(path.parent)
    Image.fromarray(image_arr).save(path, format="JPEG", quality=quality)

def zip_directory(src_dir: Path, zip_path: Path):
    ensure_dir(zip_path.parent)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for f in files:
                fp = Path(root) / f
                zf.write(fp, fp.relative_to(src_dir))

# ---------------------------
# EL preprocessing (CLAHE + blur)
# ---------------------------
def normalize_el(img_rgb: np.ndarray, clip_limit: float = 0.02, tile_size: int = 8, sigma: float = 0.6) -> np.ndarray:
    """
    EL images: cells bright, gridlines darker.
    Use CLAHE (adapthist) + mild Gaussian to enhance lines.
    """
    gray = color.rgb2gray(img_rgb)  # float [0,1]
    # CLAHE: clip_limit is fraction; typical 0.01-0.03; tile_grid via kernel_size
    clahe = exposure.equalize_adapthist(gray, clip_limit=clip_limit, kernel_size=tile_size)
    if sigma and sigma > 0:
        clahe = filters.gaussian(clahe, sigma=sigma)
    # Convert to 8-bit
    gray_u8 = util.img_as_ubyte(clahe)
    return gray_u8  # uint8

# ---------------------------
# Grid detection via morphology + projections
# ---------------------------
def binarize_image(gray_u8: np.ndarray, mode: str = "otsu") -> np.ndarray:
    if mode == "adaptive":
        # Local threshold + binarize
        block_size = 31
        offset = 5
        # Sauvola/Niblack are possible; use threshold_local for simplicity
        thr = filters.threshold_local(gray_u8, block_size=block_size, offset=offset)
        bw = (gray_u8 > thr).astype(np.uint8) * 255
    else:
        t = threshold_otsu(gray_u8)
        bw = (gray_u8 > t).astype(np.uint8) * 255
    return bw

def detect_line_maps(gray_u8: np.ndarray,
                     polarity: str = "auto",
                     binarize: str = "otsu",
                     ksize_v: int = 25,
                     ksize_h: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create vertical/horizontal line maps using erosion+dilation with rect footprints.
    In EL: lines often darker → invert after binarization.
    """
    bw = binarize_image(gray_u8, mode=binarize)  # uint8 {0,255}
    mean_val = gray_u8.mean()
    if polarity == "auto":
        use = 255 - bw if mean_val > 127 else bw
    elif polarity == "dark":
        use = 255 - bw
    else:
        use = bw

    # Convert to boolean for skimage morphology
    use_bool = use > 0

    # Vertical line enhancement: footprint height ksize_v, width 1
    vert_foot = rectangle(ksize_v, 1)
    vert = morphology.dilation(morphology.erosion(use_bool, vert_foot), vert_foot)

    # Horizontal line enhancement: footprint height 1, width ksize_h
    horiz_foot = rectangle(1, ksize_h)
    horiz = morphology.dilation(morphology.erosion(use_bool, horiz_foot), horiz_foot)

    # Back to uint8
    vert_u8 = (vert.astype(np.uint8)) * 255
    horiz_u8 = (horiz.astype(np.uint8)) * 255
    return vert_u8, horiz_u8

def project_peaks(line_map_u8: np.ndarray, axis: int = 0, min_dist: int = 30, min_strength: float = 0.3) -> List[int]:
    """
    Sum along axis and find peaks (simple local maxima with min distance).
    axis=0 → treat columns (vertical lines); axis=1 → rows (horizontal lines).
    """
    proj = line_map_u8.sum(axis=axis).astype(np.float64)
    rng = proj.max() - proj.min()
    proj_norm = (proj - proj.min()) / (rng + 1e-6)
    peaks = []
    last_idx = -min_dist
    for i in range(1, len(proj_norm) - 1):
        if proj_norm[i] > min_strength and proj_norm[i] > proj_norm[i - 1] and proj_norm[i] > proj_norm[i + 1]:
            if i - last_idx >= min_dist:
                peaks.append(i)
                last_idx = i
    return peaks

def cuts_from_peaks(peaks: List[int], maxlen: int) -> List[int]:
    if len(peaks) < 2:
        return [0, maxlen - 1]
    cuts = [0]
    for i in range(len(peaks) - 1):
        cuts.append((peaks[i] + peaks[i + 1]) // 2)
    cuts.append(maxlen - 1)
    return sorted(list(set(cuts)))

def build_cells(img_rgb: np.ndarray,
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
    for r in range(len(ycuts) - 1):
        y0, y1 = ycuts[r], ycuts[r + 1]
        for c in range(len(xcuts) - 1):
            x0, x1 = xcuts[c], xcuts[c + 1]
            w, h = x1 - x0, y1 - y0
            if w >= min_cell_w and h >= min_cell_h:
                crop = img_rgb[y0:y1, x0:x1, :].copy()
                cells.append({"row": r, "col": c, "bbox": (x0, y0, x1, y1), "image": crop})
    return cells

def draw_grid_overlay(img_rgb: np.ndarray, cells: List[Dict[str, Any]], color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
    """
    Overlay rectangles using pure NumPy (no cv2). Draw borders per cell.
    """
    vis = img_rgb.copy()
    for cell in cells:
        x0, y0, x1, y1 = cell["bbox"]
        # Top/bottom
        vis[y0:y0+thickness, x0:x1, :] = color
        vis[y1-thickness:y1, x0:x1, :] = color
        # Left/right
        vis[y0:y1, x0:x0+thickness, :] = color
        vis[y0:y1, x1-thickness:x1, :] = color
    return vis

def manual_split(img_rgb: np.ndarray, n_rows: int, n_cols: int, margin: int = 0) -> List[Dict[str, Any]]:
    H, W = img_rgb.shape[:2]
    x0, y0 = margin, margin
    x1, y1 = W - margin, H - margin
    cell_w = (x1 - x0) // n_cols
    cell_h = (y1 - y0) // n_rows
    cells = []
    for r in range(n_rows):
        for c in range(n_cols):
            cx0 = x0 + c * cell_w
            cy0 = y0 + r * cell_h
            cx1 = cx0 + cell_w
            cy1 = cy0 + cell_h
            crop = img_rgb[cy0:cy1, cx0:cx1, :].copy()
            cells.append({"row": r, "col": c, "bbox": (cx0, cy0, cx1, cy1), "image": crop})
    return cells

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="PV EL Module → Cell Segregation (no OpenCV)", layout="wide")
st.title("🔬 PV EL Module → Cell Segregation (Streamlit, scikit-image)")

st.markdown("""
Upload **EL PV module** images to automatically detect the **cell grid** and export **per‑cell crops**.
This version uses **scikit‑image** (no OpenCV), so it works smoothly on Streamlit Cloud.
If auto detection struggles, use the **Manual rows × cols** fallback.
""")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
# Preprocessing
clip_limit = st.sidebar.slider("CLAHE clip_limit (adapthist)", 0.005, 0.05, 0.02, 0.005)
tile_size = st.sidebar.slider("CLAHE tile size", 4, 32, 8, 2)
gauss_sigma = st.sidebar.slider("Gaussian sigma", 0.0, 2.0, 0.6, 0.1)

# Detection params
polarity = st.sidebar.selectbox("Line polarity", ["auto", "dark", "bright"], index=0)
binarize = st.sidebar.selectbox("Binarization", ["otsu", "adaptive"], index=0)
ksize_v = st.sidebar.slider("Vertical footprint height", 5, 75, 25, 1)
ksize_h = st.sidebar.slider("Horizontal footprint width", 5, 75, 25, 1)
min_cell_w = st.sidebar.slider("Min cell width (px)", 20, 400, 40, 10)
min_cell_h = st.sidebar.slider("Min cell height (px)", 20, 400, 40, 10)

# Manual fallback
use_manual = st.sidebar.checkbox("Use manual rows × cols fallback", False)
n_rows = st.sidebar.number_input("Rows", min_value=1, max_value=20, value=6)
n_cols = st.sidebar.number_input("Cols", min_value=1, max_value=24, value=10)
manual_margin = st.sidebar.number_input("Manual margin (px)", min_value=0, max_value=200, value=0)

# Output
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
    img_rgb = pil_to_np(img_pil)  # HWC RGB uint8

    # Preprocess EL
    gray_u8 = normalize_el(img_rgb, clip_limit=settings["clip_limit"], tile_size=settings["tile_size"], sigma=settings["gauss_sigma"])

    # Auto or manual
    if not settings["use_manual"]:
        vert_map_u8, horiz_map_u8 = detect_line_maps(
            gray_u8,
            polarity=settings["polarity"],
            binarize=settings["binarize"],
            ksize_v=settings["ksize_v"],
            ksize_h=settings["ksize_h"]
        )
        cells = build_cells(
            img_rgb,
            vert_map_u8,
            horiz_map_u8,
            min_cell_w=settings["min_cell_w"],
            min_cell_h=settings["min_cell_h"]
        )
    else:
        cells = manual_split(img_rgb, n_rows=settings["n_rows"], n_cols=settings["n_cols"], margin=settings["manual_margin"])

    overlay = draw_grid_overlay(img_rgb, cells)

    # Save outputs
    ensure_dir(save_root)
    save_image(save_root / "module_grid.jpg", overlay)
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
        "grid_overlay": overlay,
        "cells": cells,
        "elapsed": time.time() - t0
    }

# Run
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
                    "clip_limit": clip_limit,
                    "tile_size": tile_size,
                    "gauss_sigma": gauss_sigma,
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
            st.image(np_to_pil(res["grid_overlay"]), caption=f"Detected cell grid: {upl.name}", use_column_width=True)

            # Preview first 12 cells
            cols_show = st.columns(min(6, max(1, int(n_cols))))
            preview_count = min(len(res["cells"]), 12)
            for i in range(preview_count):
                r, c = res["cells"][i]["row"], res["cells"][i]["col"]
                cols_show[i % len(cols_show)].image(
                    np_to_pil(res["cells"][i]["image"]),
                    caption=f"Cell r{r} c{c}",
                    use_column_width=True
                )

            # ZIP download
            zip_path = save_dir / f"{Path(upl.name).stem}_cells.zip"
            zip_directory(save_dir, zip_path)
            with open(zip_path, "rb") as f:
                st.download_button(f"📦 Download crops for {upl.name}", data=f, file_name=zip_path.name)

st.markdown("---")
st.caption("Tips: Increase vertical/horizontal footprints when busbars are faint; switch to 'adaptive' threshold for uneven brightness; use manual rows×cols (e.g., 6×10) when needed.")
