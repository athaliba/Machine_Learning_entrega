"""Caminhos e configuração da MLP para atraso na Olist — Aula 14."""

from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

OLIST_DATA = PROJ_ROOT.parent / "olist" / "data" / "interim" / "orders_dataset_refined.csv"
SEED = 42
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 5
THRESHOLD = 0.50
