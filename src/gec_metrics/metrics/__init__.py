from .base import (
    MetricBase,
    MetricBaseForReferenceBased,
    MetricBaseForReferenceFree,
    MetricBaseForReferenceWoSource
)
from .scribendi import Scribendi
from .impara import IMPARA
from .some import SOME
from .gleu import GLEU, GLEUOfficial
from .errant import ERRANT
from .green import GREEN
from .gotoscorer import GoToScorer

METRIC_BASE_CLS = [
    MetricBase,
    MetricBaseForReferenceBased,
    MetricBaseForReferenceFree,
    MetricBaseForReferenceWoSource
]
METRIC_CLS = [
    Scribendi,
    IMPARA,
    SOME,
    GLEU,
    GLEUOfficial,
    ERRANT,
    GREEN,
    GoToScorer
]

__all__ = [c.__name__ for c in METRIC_BASE_CLS + METRIC_CLS]

METRIC_ID2CLS = {
    c.__name__.lower(): c for c in METRIC_CLS
}