import abc
from dataclasses import dataclass
from gec_metrics.metrics import (
    MetricBase,
    MetricBaseForReferenceBased,
    MetricBaseForReferenceFree,
    MetricBaseForSourceFree
)

class MetaEvalBase(abc.ABC):
    @dataclass
    class Config: ...

    @dataclass
    class Corr:
        pearson: float = None
        spearman: float = None
        accuracy: float = None
        kendall: float = None

    @dataclass
    class Output: ...

    def __init__(self, config: Config):
        self.config = config
    
    @abc.abstractmethod
    def load_system_data(self) -> dict[str, list]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def load_sentence_data(self) -> dict[str, list]:
        raise NotImplementedError
    
    def calc_system_score(
        self,
        metric: MetricBase,
        sources: list[str],
        hypotheses: list[str],
        references: list[list[str]]
    ):
        if isinstance(metric, MetricBaseForReferenceBased):
            score = metric.score_corpus(
                sources=sources,
                hypotheses=hypotheses,
                references=references
            )
        elif isinstance(metric, MetricBaseForReferenceFree):
            score = metric.score_corpus(
                sources=sources,
                hypotheses=hypotheses
            )
        elif isinstance(metric, MetricBaseForSourceFree):
            score = metric.score_corpus(
                hypotheses=hypotheses,
                references=references
            )
        return score
    
    def calc_sentence_score(
        self,
        metric: MetricBase,
        sources: list[str],
        hypotheses: list[str],
        references: list[list[str]]
    ):
        if isinstance(metric, MetricBaseForReferenceBased):
            scores = metric.score_sentence(
                sources=sources,
                hypotheses=hypotheses,
                references=references
            )
        elif isinstance(metric, MetricBaseForReferenceFree):
            scores = metric.score_sentence(
                sources=sources,
                hypotheses=hypotheses
            )
        elif isinstance(metric, MetricBaseForSourceFree):
            scores = metric.score_sentence(
                hypotheses=hypotheses,
                references=references
            )
        return scores
    
    @abc.abstractmethod
    def corr_system(self, scorer: MetricBase):
        raise NotImplementedError

    @abc.abstractmethod
    def corr_sentence(self, scorer: MetricBase):
        raise NotImplementedError