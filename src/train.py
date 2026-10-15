import io
import os
import zipfile
import joblib
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Importar función modular desde el mismo directorio src/
from preprocess import preprocess_corpus

# Configuración de rutas principales
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
)


def download_data() -> str:
  """Descarga y extrae el conjunto de datos de la UCI."""
  os.makedirs(DATA_DIR, exist_ok=True)
  file_path = os.path.join(DATA_DIR, "SMSSpamCollection")

  if not os.path.exists(file_path):
    print("📥 Descargando dataset desde UCI Repository...")
    response = requests.get(DATASET_URL)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
      z.extractall(DATA_DIR)
    print("✓ Dataset descargado y extraído correctamente.")
  else:
    print("✓ Archivo de datos localizado localmente.")

  return file_path


def run_training_pipeline():
  """Ejecuta el pipeline completo de carga, preprocesamiento y entrenamiento."""

  # 1. Carga y Limpieza Inicial
  data_path = download_data()
  df = pd.read_csv(
      data_path, sep="\t", header=None, names=["label", "message"]
  )

  initial_len = len(df)
  df = df.drop_duplicates()
  print(f"🧹 Registros únicos: {len(df)} (Eliminados {initial_len - len(df)})")

  # 2. Preprocesamiento NLP
  print("⚙️ Aplicando preprocesamiento NLP al corpus...")
  df["processed_message"] = preprocess_corpus(df["message"])
  y = df["label"].apply(lambda x: 1 if x == "spam" else 0)

  # 3. Configuración del Pipeline y Búsqueda en Grilla
  print("🚀 Configurando Pipeline y GridSearchCV...")
  pipeline = Pipeline([
      (
          "vectorizer",
          CountVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2)),
      ),
      ("classifier", MultinomialNB()),
  ])

  param_grid = {
      "classifier__alpha": [0.01, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]
  }

  grid_search = GridSearchCV(
      pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1
  )

  # 4. Ajuste del Modelo
  grid_search.fit(df["processed_message"], y)

  best_model = grid_search.best_estimator_
  print("\n=== RESULTADOS DEL ENTRENAMIENTO ===")
  print(f"• Mejores hiperparámetros : {grid_search.best_params_}")
  print(
      "• Mejor F1-Score (CV=5)  :"
      f" {grid_search.best_score_:.4f} ({grid_search.best_score_*100:.2f}%)"
  )

  # 5. Persistencia del Modelo
  os.makedirs(MODELS_DIR, exist_ok=True)
  model_output_path = os.path.join(MODELS_DIR, "spam_detection_model.joblib")
  joblib.dump(best_model, model_output_path)
  print(f"💾 Modelo serializado guardado en: '{model_output_path}'")


if __name__ == "__main__":
  run_training_pipeline()