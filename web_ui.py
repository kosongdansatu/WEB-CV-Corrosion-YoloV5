import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2
from pathlib import Path
import io

# Import fungsi-fungsi yang diperlukan dari yolov5
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import (
    check_img_size,
    non_max_suppression,
    scale_boxes,
)
from yolov5.utils.plots import Annotator, colors
from yolov5.utils.torch_utils import select_device
from yolov5.utils.dataloaders import letterbox

# --- Konfigurasi ---
WEIGHTS_PATH = "yolov5-corrosion.pt"  # Pastikan file ini ada di folder yang sama
IMG_SIZE = 640
DEVICE = ""  # Biarkan kosong untuk deteksi otomatis (CPU/GPU)

st.set_page_config(page_title="Deteksi Korosi YOLOv5", layout="wide")


# --- Fungsi untuk Memuat Model ---
@st.cache_resource  # Cache model agar tidak di-load ulang setiap kali ada interaksi
def load_yolo_model(weights, device):
    """Memuat model YOLOv5 dari file weights."""
    selected_device = select_device(device)
    model = DetectMultiBackend(
        weights, device=selected_device, dnn=False, data=None, fp16=False
    )
    return model


def run_inference(model, image, conf_thres, iou_thres):
    """Menjalankan inferensi pada gambar."""
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((IMG_SIZE, IMG_SIZE), s=stride)

    # Konversi PIL Image ke format OpenCV (numpy array)
    im0 = np.array(image)
    im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)

    # Pre-processing gambar
    img = letterbox(im0, imgsz, stride=stride, auto=pt)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    im = torch.from_numpy(img).to(model.device)
    im = im.float()
    im /= 255.0  # Normalisasi
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # Inferensi
    pred = model(im, augment=False, visualize=False)

    # Non-Maximum Suppression (NMS)
    pred = non_max_suppression(
        pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=1000
    )

    # Proses deteksi
    det = pred[0]
    annotator = Annotator(im0.copy(), line_width=3, example=str(names))

    detection_summary = {}

    if len(det):
        # Rescale boxes dari img_size ke ukuran im0
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

        # Hitung dan tampilkan hasil
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # jumlah deteksi per kelas
            detection_summary[names[int(c)]] = int(n)

        # Gambar bounding box
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)
            label = f"{names[c]} {conf:.2f}"
            annotator.box_label(xyxy, label, color=colors(c, True))

    im_result = annotator.result()
    return cv2.cvtColor(im_result, cv2.COLOR_BGR2RGB), detection_summary


# --- Tampilan Utama Aplikasi ---
def main():
    st.title("Deteksi Korosi dengan YOLOv5")
    st.write("Unggah gambar untuk mendeteksi area korosi menggunakan model YOLOv5.")

    # --- Sidebar untuk Konfigurasi ---
    st.sidebar.header("Konfigurasi")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 0.0, 1.0, 0.25, 0.05
    )
    iou_threshold = st.sidebar.slider("IOU Threshold", 0.0, 1.0, 0.45, 0.05)

    # --- Unggah Gambar ---
    uploaded_file = st.file_uploader(
        "Pilih sebuah gambar...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Gambar Asli", use_container_width=True)

        if st.button("Deteksi Korosi"):
            with st.spinner("Sedang memproses..."):
                model = load_yolo_model(WEIGHTS_PATH, DEVICE)
                result_image, summary = run_inference(
                    model, image, confidence_threshold, iou_threshold
                )

            with col2:
                st.image(
                    result_image, caption="Hasil Deteksi", use_container_width=True
                )

            st.success("Deteksi selesai!")
            st.write("Ringkasan Deteksi:", summary)


if __name__ == "__main__":
    main()
