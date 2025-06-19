# app.py
import io, datetime
import streamlit as st
from PIL import Image
from googletrans import Translator

from utils.predictor import predict_skin_disease, CLASS_NAMES
from utils.explanations import get_explanation
from utils.gradcam import generate_gradcam

# ── UI config ──────────────────────────────────────────────────
st.set_page_config(page_title="AI MedBot", page_icon="🧑‍⚕️", layout="centered")
st.title("🧑‍⚕️ AI MedBot | Skin-Disease Detector")
st.markdown(
    "Upload a skin-condition image and let AI predict the disease, "
    "explain it, and visualise where the model focused."
)

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    lang_choice = st.selectbox(
        "🌍 Output Language", ("English", "Hindi", "Spanish", "French")
    )

translator = Translator()

# ── Upload ─────────────────────────────────────────────────────
file = st.file_uploader("📤 Upload JPG / PNG image", ["jpg", "jpeg", "png"])
if not file:
    st.warning("👆 Please upload an image to start.")
    st.stop()

orig_img = Image.open(file)
st.image(orig_img, caption="📷 Uploaded Image", use_container_width=True)

if not st.button("🔍 Diagnose"):
    st.stop()

# ── Inference ─────────────────────────────────────────────────
with st.spinner("Analyzing …"):
    disease, conf = predict_skin_disease(orig_img)
    explanation = get_explanation(disease)
    if lang_choice != "English":
        explanation = translator.translate(
            explanation, dest=lang_choice[:2].lower()
        ).text
    heat_img = generate_gradcam(orig_img, class_index=CLASS_NAMES.index(disease))

# ── Results ───────────────────────────────────────────────────
st.success("✅ Diagnosis complete!")
st.markdown(f"### 🩺 **Predicted Disease:** `{disease}`")
st.markdown(f"### 📊 **Confidence:** `{conf:.2f}`")
st.markdown("### 📄 **Explanation & Care Tips:**")
st.info(explanation)

st.markdown("### 🔥 Model-Focus Heatmap")
st.image(heat_img, caption="Grad-CAM", use_container_width=True)

# ── Heat-map PNG download ─────────────────────────────────────
buf_png = io.BytesIO()
heat_img.save(buf_png, format="PNG")
st.download_button(
    "📥 Download Heatmap PNG",
    buf_png.getvalue(),
    file_name="gradcam_heatmap.png",
    mime="image/png",
)
