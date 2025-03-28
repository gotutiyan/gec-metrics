from .base import (
    MetricBase,
    MetricBaseForReferenceBased,
    MetricBaseForReferenceFree,
    MetricBaseForSourceFree,
    inputs_handler
)
from .scribendi import Scribendi
from .impara import IMPARA
from .some import SOME
from .gleu import GLEU, GLEUOfficial
from .errant import ERRANT
from .green import GREEN
from .gotoscorer import GoToScorer
from .bertscore import BertScore
from .pt_errant import PTERRANT
from .llm_kobayashi24 import (
    LLMKobayashi24HFEdit,
    LLMKobayashi24HFSent,
    LLMKobayashi24OpenAIEdit,
    LLMKobayashi24OpenAISent
)

METRIC_BASE_CLS = [
    MetricBase,
    MetricBaseForReferenceBased,
    MetricBaseForReferenceFree,
    MetricBaseForSourceFree
]
METRIC_CLS = [
    Scribendi,
    IMPARA,
    SOME,
    GLEU,
    GLEUOfficial,
    ERRANT,
    GREEN,
    GoToScorer,
    BertScore,
    PTERRANT,
    LLMKobayashi24HFEdit,
    LLMKobayashi24HFSent,
    LLMKobayashi24OpenAIEdit,
    LLMKobayashi24OpenAISent
]

__all__ = [c.__name__ for c in METRIC_BASE_CLS + METRIC_CLS]

METRIC_ID2CLS = {
    c.__name__.lower(): c for c in METRIC_CLS
}