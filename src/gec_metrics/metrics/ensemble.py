import abc
from .base import (
    MetricBase,
    MetricBaseForReferenceBased,
    inputs_handler
)
import numpy as np
from dataclasses import dataclass

class MetricEnsemble(MetricBaseForReferenceBased, metaclass=abc.ABCMeta):
    '''Class to ensemble metrics.
    '''
    @dataclass
    class Config(MetricBaseForReferenceBased.Config):
        metrics: list[MetricBase] = None

    def score_sentence(
        self,
        sources: list[str],
        hypotheses: list[str],
        references: list[list[str]]
    ) -> list[float]:
        '''Calculate a sentence-level score.

        Args:
            sources (list[str]): Source sentence.
            hypothesis (list[str]): Corrected sentences.
            references (list[list[str]]): Reference sentences.
                The shape is (the number of references, the number of sentences).
        
        Returns:
            list[float]: The sentence-level scores.
        '''
        scores = np.array([
            metric.score_sentence(
                **inputs_handler(
                    metric, sources, hypotheses, references
                )
            ) for metric in self.config.metrics
        ])  # (num_metrics, num_sents)
        ens_scores = scores.mean(axis=0)
        return list(ens_scores)
        
    def score_pairwise(
        self,
        sources: list[str],
        hypotheses: list[list[str]],
        references: list[list[str]]
    ) -> list[list[list[int]]]:
        '''Calculate pairwise scores for all of combinations of hypotheses.
        By default, it simply compares the sentence-level scores.

        Args:
            sources (list[str]): Source sentence.
            hypothesis_list (list[list[str]]): Corrected sentences.
                The shape is (num_systems, num_sentences).
            references (list[list[str]]): Reference sentences.
                The shape is (num_references, num_sentences).
        
        Returns:
            list[list[list]]: Pairwise comparison resutls.
        '''
        # (num_metrics, num_sents, num_systems, num_systems)
        scores = np.array([
            metric.score_pairwise(
                **inputs_handler(metric, sources, hypotheses, references)
            ) for metric in self.config.metrics
        ])
        # (num_sents, num_systems, num_systems)
        ens_scores = scores.sum(axis=0)
        print(ens_scores[0])
        # Majority voting
        ens_scores[ens_scores < 0] = -1
        ens_scores[ens_scores > 0] = 1
        print(ens_scores[0])
        return list(ens_scores)