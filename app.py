import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from datetime import datetime
import pandas as pd
import io
import re

# =========================
# KONFIGURASI DASAR
# =========================
IMG_SIZE = (224, 224)

class_names = [
    "Abyssunian",
    "Bengal",
    "Birman",
    "Bombay",
    "British_shorthair",
    "Egyptian_mau",
    "Maine_coon",
    "Persian",
    "Ragdoll",
    "Russian_blue",
    "Siamese",
    "Sphynx"
]

# =========================
# FUNGSI LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model_path = r"D:\!Kuliah\JST\JST\Model\kucing_resnet_model_revisi2_Old.h5"
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

# =========================
# PREPROCESSING GAMBAR
# =========================
def preprocess_image(uploaded_image):
    img = Image.open(uploaded_image).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1,224,224,3)
    return img, img_array

# =========================
# PREDIKSI
# =========================
def predict_class(img_array, model):
    preds = model.predict(img_array, verbose=0)
    probs = preds[0]
    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    pred_prob = float(probs[pred_idx])
    return pred_class, pred_prob, probs

# =========================
# (OPSIONAL) TEBAK GT DARI NAMA FILE
# Misal file: "Bengal_001.jpg" atau "bengal-12.png"
# =========================
def guess_gt_from_filename(filename: str):
    name = filename.lower()
    # normalisasi: spasi/dash jadi underscore
    name = re.sub(r"[\s\-]+", "_", name)
    for cls in class_names:
        if cls.lower() in name:
            return cls
    return None

# =========================
# HISTORY (RIWAYAT) PER SESI
# =========================
def init_history():
    if "history" not in st.session_state:
        st.session_state.history = []

def add_to_history(filename, gt_label, pred_class, pred_prob, probs, img_pil):
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    idxs = np.argsort(probs)[::-1][:3]
    top3_text = ", ".join([f"{class_names[i]} ({probs[i]*100:.1f}%)" for i in idxs])

    is_correct = None
    if gt_label and gt_label != "— (tidak diisi)":
        is_correct = (gt_label == pred_class)

    st.session_state.history.append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "gt": gt_label,
        "pred_class": pred_class,
        "confidence": pred_prob,
        "correct": is_correct,
        "top3": top3_text,
        "img_bytes": img_bytes
    })

def history_to_df():
    rows = []
    for h in st.session_state.history:
        rows.append({
            "time": h["time"],
            "filename": h["filename"],
            "gt": h["gt"],
            "pred_class": h["pred_class"],
            "confidence_%": round(h["confidence"] * 100, 2),
            "correct": h["correct"],
            "top3": h["top3"]
        })
    return pd.DataFrame(rows)

# =========================
# STREAMLIT UI
# =========================
def main():
    st.set_page_config(page_title="Klasifikasi Ras Kucing", page_icon="🐱")
    init_history()

    st.title("🐱 Klasifikasi Ras Kucing dengan ResNet50V2")
    st.write(
        """
        Upload foto kucing, lalu model akan memprediksi ras kucing tersebut.
        Kamu juga bisa isi **Ground Truth (GT)** supaya muncul perbandingan GT vs Prediksi.
        """
    )

    # Sidebar
    st.sidebar.title("Informasi")
    st.sidebar.markdown(
        """
        **Cara pakai:**
        1. Upload gambar kucing.
        2. (Opsional) Pilih Ground Truth (GT).
        3. Lihat hasil prediksi + history.

        Pastikan:
        - Path model `.h5` sudah benar di `load_model()`.
        - `class_names` sesuai urutan kelas waktu training.
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("History")
    show_thumbs = st.sidebar.checkbox("Tampilkan thumbnail history", value=True)

    if st.sidebar.button("🧹 Clear history"):
        st.session_state.history = []
        st.sidebar.success("History dibersihkan.")

    # Upload gambar
    uploaded_file = st.file_uploader(
        "Upload gambar kucing (jpg, jpeg, png):",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.subheader("Gambar yang di-upload")

        img, img_array = preprocess_image(uploaded_file)
        st.image(img, caption="Gambar input", use_container_width=True)

        # GT input
        filename = getattr(uploaded_file, "name", "uploaded_image")
        guessed = guess_gt_from_filename(filename)

        gt_options = ["— (tidak diisi)"] + class_names
        default_gt = guessed if guessed else "— (tidak diisi)"
        gt_index = gt_options.index(default_gt)

        gt_label = st.selectbox(
            "Ground Truth (GT) (opsional):",
            options=gt_options,
            index=gt_index
        )

        # Prediksi
        with st.spinner("Memuat model dan melakukan prediksi..."):
            model = load_model()
            pred_class, pred_prob, probs = predict_class(img_array, model)

        # Tampilkan hasil
        st.subheader("Hasil Prediksi")
        st.markdown(
            f"**GT:** `{gt_label}`  \n"
            f"**Prediksi:** `{pred_class}`  \n"
            f"**Probabilitas:** `{pred_prob * 100:.2f}%`"
        )

        if gt_label != "— (tidak diisi)":
            if gt_label == pred_class:
                st.success("✅ Prediksi BENAR (sesuai GT)")
            else:
                st.error("❌ Prediksi SALAH (tidak sesuai GT)")

        # Probabilitas per kelas
        st.subheader("Probabilitas per Kelas")
        prob_dict = {cls: float(p) for cls, p in zip(class_names, probs)}
        prob_dict_sorted = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))

        st.write(prob_dict_sorted)
        st.bar_chart(prob_dict_sorted)

        # Tambah ke history
        add_to_history(
            filename=filename,
            gt_label=gt_label,
            pred_class=pred_class,
            pred_prob=pred_prob,
            probs=probs,
            img_pil=img
        )

    # =========================
    # HISTORY
    # =========================
    st.markdown("---")
    st.header("📜 History Prediksi (Sesi Ini)")

    if len(st.session_state.history) == 0:
        st.info("Belum ada history. Upload gambar untuk mulai.")
        return

    df = history_to_df()
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "⬇️ Download history (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="history_prediksi_kucing.csv",
        mime="text/csv"
    )

    if show_thumbs:
        with st.expander("Lihat thumbnail history (10 terakhir)"):
            for h in reversed(st.session_state.history[-10:]):
                cols = st.columns([1, 4])
                with cols[0]:
                    st.image(Image.open(io.BytesIO(h["img_bytes"])), use_container_width=True)
                with cols[1]:
                    status = ""
                    if h["correct"] is True:
                        status = "✅ BENAR"
                    elif h["correct"] is False:
                        status = "❌ SALAH"
                    st.markdown(
                        f"**GT:** {h['gt']} | **Pred:** {h['pred_class']} ({h['confidence']*100:.2f}%) {status}  \n"
                        f"{h['top3']}  \n"
                        f"🕒 {h['time']} | 📄 {h['filename']}"
                    )

if __name__ == "__main__":
    main()
