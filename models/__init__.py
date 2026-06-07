from .baseline import BaselineBias
from .knn_cf import ItemKNN
from .als import ALS
from .svd import FunkSVD

MODELS = {
    "baseline": BaselineBias,
    "itemknn": ItemKNN,
    "als": ALS,
    "svd": FunkSVD
}