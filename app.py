import os
import joblib
import streamlit as st
from src.preprocess import preprocess_message

# Configuración de página
st.set_page_config(
    page_title="SMS Spam Detector", page_icon="🛡️", layout="centered"
)

MODEL_PATH = os.path.join("models", "spam_detection_model.joblib")


@st.cache_resource
def load_pipeline():
  if not os.path.exists(MODEL_PATH):
    st.error(
        "No se encontró el archivo del modelo. Ejecuta primero `python"
        " src/train.py`."
    )
    st.stop()
  return joblib.load(MODEL_PATH)


pipeline = load_pipeline()

st.title("🛡️ SMS Spam Detector")
st.write(
    "Detección de mensajes no deseados utilizando **Multinomial Naive Bayes**"
    " y **NLP**."
)

user_input = st.text_area(
    "Introduce el SMS a analizar:",
    placeholder="Ejemplo: Free entry in a weekly competition to win an iPad...",
)

if st.button("Clasificar Mensaje", type="primary"):
  if user_input.strip():
    processed_input = preprocess_message(user_input)
    prediction = pipeline.predict([processed_input])[0]
    probabilities = pipeline.predict_proba([processed_input])[0]

    spam_prob = probabilities[1]
    ham_prob = probabilities[0]

    st.markdown("---")
    if prediction == 1:
      st.error(f"🚨 **Resultado: SPAM** (Confianza: {spam_prob:.2%})")
    else:
      st.success(f"✅ **Resultado: HAM / LEGÍTIMO** (Confianza: {ham_prob:.2%})")

    st.caption(f"Texto procesado por el pipeline: `{processed_input}`")
  else:
    st.warning("Por favor, introduce un texto antes de clasificar.")