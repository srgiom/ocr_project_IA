import string
from pathlib import Path

# Caracteres soportados
PRINTED_CHARS = string.ascii_uppercase + string.ascii_lowercase + string.digits + ".,:;!?()/-'\" "

HANDWRITTEN_CHARS = string.ascii_uppercase + string.digits + " "

# Tamaño normalizado de cada carácter
CHAR_SIZE = (32, 32)

# Directorios base
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Nombres de modelos
PRINTED_MODEL_PATH = MODELS_DIR / "printed_ocr_model.pkl"
HANDWRITTEN_MODEL_PATH = MODELS_DIR / "handwritten_ocr_model.pkl"

# Parámetros HOG "caseros"
HOG_N_BINS = 9
HOG_CELL_SIZE = (8, 8)   # píxeles
HOG_BLOCK_SIZE = (2, 2)  # celdas

# Segmentación
MIN_COMPONENT_AREA = 20      # área mínima para considerar un componente como posible carácter
LINE_MERGE_THRESHOLD = 0.7   # cuánto solapan verticalmente dos componentes para ir a la misma línea
WORD_GAP_FACTOR = 1.8        # factor sobre la media de gaps para decidir si es nuevo espacio

# Otros
RANDOM_SEED = 42
