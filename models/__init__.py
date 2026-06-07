from .baseline import BaselineBias
from .knn_cf import ItemKNN
from .als import ALS

MODELS = {
    "baseline": BaselineBias,
    "itemknn": ItemKNN,
    "als": ALS,
}