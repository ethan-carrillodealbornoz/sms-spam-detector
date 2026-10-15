import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def _ensure_nltk_resources():
  """Asegura la disponibilidad de las dependencias de NLTK."""
  resources = ["punkt", "punkt_tab", "stopwords"]
  for resource in resources:
    try:
      nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
      nltk.download(resource, quiet=True)


# Inicializar recursos a nivel de módulo
_ensure_nltk_resources()
_STOP_WORDS = set(stopwords.words("english"))
_STEMMER = PorterStemmer()


def preprocess_message(message: str) -> str:
  """Aplica el pipeline de preprocesamiento NLP a un string individual.

  1. Conversión a minúsculas (Lowercasing)
  2. Limpieza Regex (conserva letras, espacios, $ y !)
  3. Tokenización (word_tokenize)
  4. Eliminación de Stop Words (Inglés)
  5. Derivación regresiva (Porter Stemmer)
  6. Recomposición a cadena de texto

  Parameters:
      message (str): Mensaje de texto en bruto.

  Returns:
      str: Mensaje procesado listo para vectorización.
  """
  if not isinstance(message, str):
    message = str(message)

  # 1. Lowercasing
  message = message.lower()

  # 2. Filtrado Regex (conservando $ y !)
  message = re.sub(r"[^a-z\s$!]", "", message)

  # 3. Tokenización
  tokens = word_tokenize(message)

  # 4. Eliminación de Stop Words
  tokens = [word for word in tokens if word not in _STOP_WORDS]

  # 5. Stemming
  tokens = [_STEMMER.stem(word) for word in tokens]

  # 6. Recomposición
  return " ".join(tokens)


def preprocess_corpus(series_or_list) -> list:
  """Aplica preprocess_message sobre una colección completa de mensajes."""
  return [preprocess_message(msg) for msg in series_or_list]