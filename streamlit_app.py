
import streamlit as st
import io, json, time, zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from skimage import color, exposure, filters, morphology, util
from skimage.filters import threshold_otsu
from skimage.morphology import rectangle

st.set_page_config(page_title="EL Cell Segregation (scikit-image)", layout="wide")
st.title("🔬 EL PV Module → Cell Segregation (Streamlit, scikit-image)")

def normalize_el(img_rgb, clip_limit=0.02, tile_size=8, sigma=0.6):
    gray = color.rgb2gray(img_rgb)                # float [0,1]
    clahe = exposure.equalize_adapthist(gray, clip_limit=clip_limit, kernel_size=tile_size)
    if sigma and sigma > 0:
        clahe = filters.gaussian(clahe, sigma=sigma)
    return util.img_as_ubyte(clahe)               # uint8

def binarize_image(gray_u8, mode="otsu"):
    if mode == "adaptive":
        thr = filters.threshold_local(gray_u8, block_size=31, offset=5)
        return (gray_u8 > thr).astype(np.uint8) * 255
    t = threshold_otsu(gray_u8)
    return (gray_u8 > t).astype(np.uint8) * 255

def detect_line_maps(gray_u8, polarity="auto", binarize="otsu", ksize_v=25, ksize_h=25):
    bw = binarize_image(gray_u8, mode=binarize)
    use = (255 - bw) if (polarity == "dark" or (polarity == "auto" and gray_u8.mean() > 127)) else bw
    use_bool = use > 0
    vert = morphology.dilation(morphology.erosion(use_bool, rectangle(ksize_v, 1)), rectangle(ksize_v, 1))
    horiz = morphology.dilation(morphology.erosion(use_bool, rectangle(1, ksize_h)), rectangle(1, ksize_h))
    return (vert.astype(np.uint8) * 255), (horiz.astype(np.uint8) * 255)

def project_peaks(line_map_u8, axis=0, min_dist=30, min_strength=0.3):
    proj = line_map_u8.sum(axis=axis).astype(np.float64)
    proj_norm = (proj - proj.min()) / (proj.max() - proj.min() + 1e-6)
    peaks, last = [], -min_dist
    for i in range(1, len(proj_norm)-1):
        if proj_norm[i] > min_strength and proj_norm[i] > proj_norm[i-1] and proj_norm[i] > proj_norm[i+1]:
            if i - last >= min_dist:
                peaks.append(i); last = i
    return peaks

def cuts_from_peaks(peaks, maxlen):
    if len(peaks) < 2: return [0, maxlen-1]
    cuts = [0] + [ (peaks[i] + peaks[i+1])//2 for i in range(len(peaks)-1) ] + [maxlen-1]
    return sorted(list(set(cuts)))

def build_cells(img_rgb, vert_map_u8, horiz_map_u8, min_cell_w=40, min_cell_h=40):
    H, W = vert_map_u8.shape
    xs = project_peaks(vert_map_u8, axis=0, min_dist=max(20, W//30))
    ys = project_peaks(horiz_map_u8, axis=1, min_dist=max(20, H//30))
    xcuts = cuts_from_peaks(xs, W); ycuts = cuts_from_peaks(ys, H)
    cells = []
    for r in range(len(ycuts)-1):
        y0, y1 = ycuts[r], ycuts[r+1]
        for c in range(len(xcuts)-1):
            x0, x1 = xcuts[c], xcuts[c+1]
            w, h = x1-x0, y1-y0
            if w >= min_cell_w and h >= min_cell_h:
                crop = img_rgb[y0:y1, x0:x1, :].copy()
                cells.append({"row": r, "col": c, "bbox": (x0, y0, x1, y1), "image": crop})
    return cells

def draw_grid_overlay(img_rgb, cells, color=(0,255,0), thickness=2):
    vis = img_rgb.copy()
    for cell in cells:
        x0,y0,x1,y1 = cell["bbox"]
        vis[y0:y0+thickness, x0:x1, :] = color
        vis[y1-thickness:y1, x0:x1, :] = color
        vis[y0:y1, x0:x0+thickness, :] = color
        vis[y0:y1, x1-thickness:x1, :] = color
    return vis

uploads = st.file_uploader("Upload EL module image(s)", type=["jpg","jpeg","png","bmp","tif","tiff"], accept_multiple_files=True)
clip_limit = st.sidebar.slider("CLAHE clip_limit", 0.005, 0.05, 0.02, 0.005)
tile_size  = st.sidebar.slider("CLAHE tile size", 4, 32, 8, 2)
sigma      = st.sidebar.slider("Gaussian sigma", 0.0, 2.0, 0.6, 0.1)
polarity   = st.sidebar.selectbox("Line polarity", ["auto","dark","bright"], index=0)
binarize   = st.sidebar.selectbox("Binarization", ["otsu","adaptive"], index=0)
ks_v       = st.sidebar.slider("Vertical footprint height", 5, 75, 25, 1)
ks_h       = st.sidebar.slider("Horizontal footprint width", 5, 75, 25, 1)
min_w      = st.sidebar.slider("Min cell width (px)", 20, 400, 40, 10)
min_h      = st.sidebar.slider("Min cell height (px)", 20, 400, 40, 10)
start      = st.sidebar.button("🚀 Run")

if start:
    if not uploads:
        st.warning("Please upload at least one image.")
    else:
        for upl in uploads:
            img = Image.open(io.BytesIO(upl.read())).convert("RGB")
            rgb = np.array(img)
            t0 = time.time()
            gray_u8 = normalize_el(rgb, clip_limit=clip_limit, tile_size=tile_size, sigma=sigma)
            vert_u8, horiz_u8 = detect_line_maps(gray_u8, polarity=polarity, binarize=binarize, ksize_v=ks_v, ksize_h=ks_h)
            cells = build_cells(rgb, vert_u8, horiz_u8, min_cell_w=min_w, min_cell_h=min_h)
            overlay = draw_grid_overlay(rgb, cells)
            st.success(f"{upl.name}: {len(cells)} cells in {time.time()-t0:.2f}s")
            st.image(overlay, caption=f"Grid overlay: {upl.name}", use_column_width=True)
            cols_show = st.columns(min(6, 6))
            for i, cell in enumerate(cells[:12]):
                cols_show[i % len(cols_show)].image(cell["image"], caption=f"Cell r{cell['row']} c{cell['col']}", use_column_width=True)
