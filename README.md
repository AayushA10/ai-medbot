# 🧠 AI MedBot – Skin Disease Detection with Grad-CAM

AI MedBot is a Streamlit-powered intelligent medical assistant that uses deep learning to detect common skin diseases from images. It also provides explainable predictions via Grad-CAM heatmaps, helping users understand *why* the model predicted a particular condition.


---

## 🔍 Features

- 🧑‍⚕️ Predicts skin diseases using a trained CNN model (`skin_model.h5`)
- 🌡️ Grad-CAM visualization for model explainability
- 🖼️ User-friendly interface built with Streamlit
- 📁 Easily extendable to new datasets or models

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit, HTML/CSS
- **Backend**: Python, TensorFlow/Keras, OpenCV
- **Explainability**: Grad-CAM via NumPy + OpenCV
- **Model**: CNN trained on skin disease images

---

## 🚀 How to Run

### 1. Clone the repository
bash
git clone https://github.com/AayushA10/ai-medbot.git
cd ai-medbot

2. Set up environment
python3 -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

3. Run the app
streamlit run app.py

