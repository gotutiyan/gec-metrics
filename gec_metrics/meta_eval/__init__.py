from .base import MetaEvalBase
from .gjg import MetaEvalGJG
from .seeda import MetaEvalSEEDA

METAEVAL_BASE_CLS = [
    MetaEvalBase
]
METAEVAL_CLS = [
    MetaEvalGJG,
    MetaEvalSEEDA
]

__all__ = [c.__name__ for c in METAEVAL_BASE_CLS + METAEVAL_CLS]

METAEVAL_ID2CLS = {
    c.__name__.lower().replace('metaeval', ''): c for c in METAEVAL_CLS
}