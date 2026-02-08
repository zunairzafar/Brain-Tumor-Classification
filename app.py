import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np


# ---------------------------
# Page config + simple styling
# ---------------------------
st.set_page_config(
    page_title="Brain Tumor Detector",
    page_icon="🧠",
    layout="centered",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
      .stButton button { border-radius: 10px; padding: 0.6rem 1rem; }
      .stFileUploader { border-radius: 12px; }
      .metric-box {
        border: 1px solid rgba(49,51,63,0.2);
        border-radius: 14px;
        padding: 14px 16px;
        background: rgba(250,250,252,0.6);
      }
      .small-note { color: rgba(49,51,63,0.7); font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------
# Model loading (cached)
# ---------------------------
@st.cache_resource
def load_model(checkpoint_path: str):
    device = torch.device("cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)
    num_classes = ckpt.get("num_classes", 2)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)

    class_names = ckpt.get("class_names", None)
    if class_names is None:
        # fallback for binary
        class_names = ["No Tumor", "Tumor"] if num_classes == 2 else [f"Class {i}" for i in range(num_classes)]

    return model, class_names


# ---------------------------
# Preprocessing (must match training)
# ---------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


@torch.no_grad()
def predict(model, pil_img: Image.Image, class_names):
    device = torch.device("cpu")

    x = preprocess(pil_img.convert("RGB")).unsqueeze(0).to(device)  # (1,3,224,224)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]
    confidence = float(probs[pred_idx])
    return pred_idx, pred_label, confidence, probs


# ---------------------------
# UI
# ---------------------------
st.title("🧠 Brain Tumor Detector")
st.write("Upload an MRI image. The model predicts **Tumor** or **No Tumor** with confidence.")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Decision threshold (Tumor)", 0.0, 1.0, 0.50, 0.01)
    st.caption("For binary classification: if P(Tumor) ≥ threshold → Tumor.")

checkpoint_path = "models/resnet18_brain_tumor.pt"
model, class_names = load_model(checkpoint_path)

uploaded = st.file_uploader("Upload an image (PNG/JPG/JPEG)", type=["png", "jpg", "jpeg"])

if uploaded is None:
    st.info("Please upload an image to run prediction.")
    st.stop()

img = Image.open(uploaded)

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Input")
    st.image(img, use_container_width=True)

with col2:
    st.subheader("Prediction")

    pred_idx, pred_label, conf, probs = predict(model, img, class_names)

    # If binary, apply threshold to tumor probability (assume index 1 is Tumor)
    if len(class_names) == 2:
        tumor_prob = float(probs[1])
        final_label = "Tumor" if tumor_prob >= threshold else "No Tumor"
        final_conf = tumor_prob if final_label == "Tumor" else 1.0 - tumor_prob

        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Result", final_label)
        st.metric("Confidence", f"{final_conf*100:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.progress(min(max(tumor_prob, 0.0), 1.0))
        st.write(f"**P(Tumor):** {tumor_prob:.4f}")
        st.write(f"**P(No Tumor):** {float(probs[0]):.4f}")

    else:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Result", pred_label)
        st.metric("Confidence", f"{conf*100:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("Class probabilities:")
        for i, name in enumerate(class_names):
            st.write(f"- {name}: {float(probs[i]):.4f}")

st.markdown('<p class="small-note">Note: This is a research/demo tool, not a medical diagnosis.</p>', unsafe_allow_html=True)
