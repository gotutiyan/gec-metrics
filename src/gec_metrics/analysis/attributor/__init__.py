from .base import AttributorBase
from .add import AttributorAdd
from .sub import AttributorSub
from .shapley import AttributorShapley
from .shapley_sampling import AttributorShapleySampling

__all__ = [
    "AttributorBase",
    "AttributorAdd",
    "AttributorSub",
    "AttributorShapley",
    "AttributorShapleySampling",
]

CLS = [
    AttributorAdd,
    AttributorSub,
    AttributorShapley,
    AttributorShapleySampling
]

NAME2CLS = {
    c.__name__.lower().replace('attributor', ''): c for c in CLS
}

def get_attributor_ids():
    return sorted(list(NAME2CLS.keys()))

def get_attributor(name: str):
    assert name in get_attributor_ids()
    return NAME2CLS[name]
    