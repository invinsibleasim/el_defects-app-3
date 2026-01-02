
import streamlit as st

st.title("Streamlit Cloud environment health check")
st.write("If this runs, your dependency install was successful.")

try:
    import cv2, numpy as np, PIL
    st.success(f"✅ OpenCV: {cv2.__version__}")
    st.success(f"✅ NumPy: {np.__version__}")
    st.success(f"✅ Pillow: {PIL.__version__}")
except Exception as e:
    st.error(f"❌ Import error: {e}")
    st.stop()

st.image(np.zeros((100,100,3), dtype=np.uint8), caption="OpenCV & NumPy basic image OK")
