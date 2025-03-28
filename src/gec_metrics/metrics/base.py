import abc
from dataclasses import dataclass
import itertools
from gecommon import apply_edits
import numpy as np
import trueskill

class MetricBase(abc.ABC):
    @dataclass
    class Config: ...

    def __init__(self, config: Config = None):
        self.config = config if config is not None else self.Config()
        self.apply_edits = apply_edits

    def make_pairwise_scores(
        self,
        scores: list[list[float]]
    ) -> list[list[list]]:
        '''Convert sentence-level scores into pairwise comparison results.
        
        Args:
            scores (list[list[float]]): Sentence-level score.
                The shape is (num_systems, num_sentences).
        
        Returns:
            list[list[list]]: Pairwise comparison resutls
                for all of combination of the systems.
                The shape is (num_sents, num_systems, num_systems).
                You can refer to the comparison result by [sent_id][sys_id1][sys_id2].
                Each element is -1, 0, or 1:
                    0 : tie
                    1 : sys_id1 wins sys_id2
                    -1: sys_id1 loses sys_id2
        '''
        num_sys = len(scores)
        num_sents = len(scores[0])
        pairwise_scores = []
        for sent_id in range(num_sents):
            # Extract `sent_id`-th scores.
            this_scores = [
                scores[sys_id][sent_id] for sys_id in range(num_sys)
            ]
            # (num_sys, num_sys)
            pairwise_table = [[0 for _ in range(num_sys)] for _ in range(num_sys)]
            for i1, i2 in itertools.combinations(range(num_sys), 2):
                judge = 0
                if this_scores[i1] > this_scores[i2]:
                    judge = 1
                elif this_scores[i1] < this_scores[i2]:
                    judge = -1
                pairwise_table[i1][i2] = judge
                pairwise_table[i2][i1] = -judge
            pairwise_scores.append(pairwise_table)
        return pairwise_scores
    
    def run_trueskill(
        self,
        pairwise_scores: list[list[list[int]]]
    ) -> list[float]:
        '''Apply TrueSkill given pairwise comparison scores.

        Args:
            pairwise_scores (list[list[list[int]]]): Pairwise comparison results.
                The shape is (num_sents, num_systems, num_systems).
                
        Returns:
            list[float]: System-level scores.
        '''
        env = trueskill.TrueSkill(
            mu=0.0,
            sigma=0.5,
            beta=0.25,
            tau=0.0,
            draw_probability=0.25
        )
        env.make_as_global()
        num_sys = len(pairwise_scores[0])
        # Temporality create system name, "0", "1", ...
        system_names = [f'{i}' for i in range(num_sys)]
        players = [
            {m: trueskill.Rating()} for m in system_names
        ]
        num_sys = len(system_names)
        num_sents = len(pairwise_scores)
        for sent_id in range(num_sents):
            ids = list(range(num_sys))
            for i1, i2 in itertools.combinations(ids, 2):
                if pairwise_scores[sent_id][i1][i2] is None:
                    continue
                if pairwise_scores[sent_id][i1][i2] == 1:
                    this_rank = (0, 1)  # smaller rank is better
                elif pairwise_scores[sent_id][i1][i2] == -1:
                    this_rank = (1, 0)
                else:
                    this_rank = (0, 0)
                players[i1], players[i2] = env.rate(
                    (players[i1], players[i2]),
                    ranks=this_rank
                )
        final_scores = [
            players[i][sys_name].mu for i, sys_name in enumerate(system_names)
        ]
        return final_scores
    

class MetricBaseForReferenceBased(MetricBase, metaclass=abc.ABCMeta):
    '''Abstract class for refernece-based metrics.
    All reference-based metrics must be implemented by inheriting from this class.
    '''
    @dataclass
    class Config(MetricBase.Config): ...

    class Score:
        '''Handle edit or n-gram count.
        - tp: True Positive.
        - fp: False Positive.
        - fn: False Negative
        - tn: True Negative.
        - beta: The beta for F-beta score. 
        '''
        def __init__(
            self,
            tp: float=0.0,
            fp: float=0.0,
            fn: float=0.0,
            tn: float=0.0,
            beta: float=0.5
        ):
            self.tp: float = tp
            self.fp: float = fp
            self.fn: float = fn
            self.tn: float = tn
            self.beta: float = beta

        def __add__(self, other) -> "Score":
            '''This overloads + operation.'''
            return self.__class__(
                tp=self.tp + other.tp,
                fp=self.fp + other.fp,
                fn=self.fn + other.fn,
                tn=self.tn + other.tn,
                beta=self.beta
            )
        
        def __lt__(self, other):
            '''This overloads < operation.
                We first compare F-score, then compare tp, then fp, finally fn.
            '''
            return [self.f, self.tp, -self.fp, -self.fn] \
                < [other.f, other.tp, -other.fp, -other.fn]

        @property
        def precision(self) -> float:
            '''Calculate the precision.'''
            if self.fp == 0:
                return 1.0
            return self.tp / (self.tp + self.fp)
        
        @property
        def recall(self) -> float:
            '''Calculate the recall '''
            if self.fn == 0:
                return 1.0
            return self.tp / (self.tp + self.fn)

        @property
        def f(self) -> float:
            '''Calculate the F-beta score. '''
            p = self.precision
            r = self.recall
            beta = self.beta
            f = float((1+(beta**2))*p*r)/(((beta**2)*p)+r) if p+r else 0.0
            return f
        
        @property
        def accuracy(self) -> float:
            '''Calculate the accuracy.'''
            if self.tp + self.fp + self.fn + self.tn == 0:
                return 0
            else:
                return (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn) 
        
        def __repr__(self):
            '''This call when you use print() method.'''
            return f"F-{self.beta}={self.f}\n Prec={self.precision}\n Rec={self.recall}\n TP={self.tp}, FP={self.fp}, FN={self.fn}, TN={self.tn}\n"

    def score_corpus(
        self,
        sources: list[str],
        hypotheses: list[str],
        references: list[list[str]]
    ) -> float:
        '''Calculate a corpus-level score.
        By default, we use the average of the sentence-level scores.

        Args:
            sources (list[str]): Source sentence.
                The shape is (num_sentences, )
            hypotheses (list[str]): Corrected sentences.
                The shape is (num_sentences, )
            references (list[list[str]]): Reference sentences.
                The shape is (num_references, num_sentences).
        
        Returns:
            float: The corpus-level score.
        '''
        scores = self.score_sentence(
            sources=sources,
            hypotheses=hypotheses,
            references=references
        )
        return sum(scores) / len(scores)
        
        
    @abc.abstractmethod
    def score_sentence(
        self,
        sources: list[str],
        hypotheses: list[str],
        references: list[list[str]]
    ) -> list[float]:
        '''Calculate sentence-level scores.

        Args:
            sources (list[str]): Source sentence.
                The shape is (num_sentences, )
            hypotheses (list[str]): Corrected sentences.
                The shape is (num_sentences, )
            references (list[list[str]]): Reference sentences.
                The shape is (num_references, num_sentences).
        
        Returns:
            list[float]: The sentence-level scores.
        '''
        raise NotImplementedError
    
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
                The shape is (num_sentences, )
            hypotheses (list[list[str]]): Corrected sentences.
                The shape is (num_systems, num_sentences).
            references (list[list[str]]): Reference sentences.
                The shape is (num_references, num_sentences).
        
        Returns:
            list[list[list]]: Pairwise comparison resutls.
                The shape is (num_sentences, num_systems, num_systems).
        '''
        scores = [
            self.score_sentence(
                sources, hyps, references
            ) for hyps in hypotheses
        ]  # (num_systems, num_sentences)
        return self.make_pairwise_scores(scores)
    
    def rank_systems(
        self,
        sources: list[str],
        hypotheses: list[list[str]],
        references: list[list[str]],
        aggregation='default'
    ) -> list[float]:
        '''Compute ranking score for multiple systems.

        Args:
            sources (list[str]): Source sentence.
                The shape is (num_sentences, )
            hypotheses (list[list[str]]): Corrected sentences.
                The shape is (num_systems, num_sentences).
            references (list[list[str]]): Reference sentences.
                The shape is (num_references, num_sentences).
            aggregation: (str): How to aggregate sentence-level scores.
                - "default" follows an original aggregation, e.g., average or accumulation.
                - "trueskill" convert sentence-level scores into pairwise comparison results,
                    then apply TrueSkill. This is motivated by https://arxiv.org/abs/2502.09416.
        
        Retunrns:
            list[float]: System-level scores.
        '''
        if aggregation == "default":
            scores = []
            for hyp_id, hyps in enumerate(hypotheses):
                scores.append(
                    self.score_corpus(
                        sources, hyps, references
                    )
                )
        elif aggregation == 'trueskill':
            pairwise_scores = self.score_pairwise(
                sources, hypotheses, references
            )
            scores = self.run_trueskill(pairwise_scores)
        return scores
    
    
class MetricBaseForReferenceFree(MetricBase, metaclass=abc.ABCMeta):
    @dataclass
    class Config(MetricBase.Config): ...

    def score_corpus(
        self,
        sources: list[str],
        hypotheses: list[str]
    ) -> float:
        '''Calculate a corpus-level score.
        By default, we use the average of the sentence-level scores.

        Args:
            sources (list[str]): Source sentence.
                The shape is (num_sentences, )
            hypotheses (list[str]): Corrected sentences.
                The shape is (num_sentences, )
        
        Returns:
            float: The corpus-level score.
        '''
        scores = self.score_sentence(
            sources=sources,
            hypotheses=hypotheses
        )
        return sum(scores) / len(scores)
        
        
    @abc.abstractmethod
    def score_sentence(
        self,
        sources: list[str],
        hypotheses: list[str]
    ) -> list[float]:
        '''Calculate a sentence-level scores.

        Args:
            sources (list[str]): Source sentence.
                The shape is (num_sentences, )
            hypotheses (list[str]): Corrected sentences.
                The shape is (num_sentences, )
        
        Returns:
            list[float]: The sentence-level scores.
        '''
        raise NotImplementedError
    
    def score_pairwise(
        self,
        sources: list[str],
        hypotheses: list[list[str]]
    ) -> list[list[list[int]]]:
        '''Calculate pairwise scores for all of combinations of hypotheses.
        By default, it simply compares the sentence-level scores.

        Args:
            sources (list[str]): Source sentence.
                The shape is (num_sentences, )
            hypotheses (list[list[str]]): Corrected sentences.
                The shape is (num_systems, num_sentences).
        
        Returns:
            list[list[list]]: Pairwise comparison resutls.
                The shape is (num_sentences, num_systems, num_systems).
        '''
        scores = [
            self.score_sentence(
                sources, hyps
            ) for hyps in hypotheses
        ]  # (num_systems, num_sentences)
        return self.make_pairwise_scores(scores)
    
    def rank_systems(
        self,
        sources: list[str],
        hypotheses: list[list[str]],
        aggregation='default'
    ):
        '''Compute ranking score for multiple systems.

        Args:
            sources (list[str]): Source sentence.
                The shape is (num_sentences, )
            hypotheses (list[list[str]]): Corrected sentences.
                The shape is (num_systems, num_sentences).
            aggregation: (str): How to aggregate sentence-level scores.
                - "default" follows an original aggregation, e.g., average or accumulation.
                - "trueskill" convert sentence-level scores into pairwise comparison results,
                    then apply TrueSkill. This is motivated by https://arxiv.org/abs/2502.09416.
        
        Retunrns:
            list[float]: System-level scores.
        '''
        if aggregation == "default":
            scores = [
                self.score_corpus(
                    sources, hyps
                ) for hyps in hypotheses
            ]  # (num_systems, num_sentences)
        elif aggregation == 'trueskill':
            pairwise_scores = self.score_pairwise(
                sources, hypotheses
            )
            scores = self.run_trueskill(pairwise_scores)
        return scores
    
class MetricBaseForSourceFree(MetricBase, metaclass=abc.ABCMeta):
    '''Metric without source sentence.
        This is basically for BERTScore or BARTScore 
            (that will be a component of PT-{ERRANT, M2}.).
    '''
    @dataclass
    class Config(MetricBase.Config): ...

    def score_corpus(
        self,
        hypotheses: list[str],
        references: list[list[str]]
    ) -> float:
        '''Calculate a corpus-level score.
        By default, we use the average of the sentence-level scores.

        Args:
            hypotheses (list[str]): Corrected sentences.
                The shape is (num_sentences, )
            references (list[list[str]]): Reference sentences.
                The shape is (num_references, num_sentences).
        
        Returns:
            float: The corpus-level score.
        '''
        scores = self.score_sentence(
            hypotheses=hypotheses,
            references=references
        )
        return sum(scores) / len(scores)
        
        
    @abc.abstractmethod
    def score_sentence(
        self,
        hypotheses: list[str],
        references: list[list[str]]
    ) -> list[float]:
        '''Calculate a sentence-level scores.

        Args:
            hypotheses (list[str]): Corrected sentences.
                The shape is (num_sentences, )
            references (list[list[str]]): Reference sentences.
                The shape is (num_references, num_sentences).
        
        Returns:
            list[float]: The sentence-level scores.
        '''
        raise NotImplementedError
    
    def score_pairwise(
        self,
        hypotheses: list[list[str]],
        references: list[list[str]]
    ) -> list[list[list[int]]]:
        '''Calculate pairwise scores for all of combinations of hypotheses.
        By default, it simply compares the sentence-level scores.

        Args:
            hypotheses (list[list[str]]): Corrected sentences.
                The shape is (num_systems, num_sentences).
            references (list[list[str]]): Reference sentences.
                The shape is (num_references, num_sentences).
        
        Returns:
            list[list[list]]: Pairwise comparison resutls.
                The shape is (num_sentences, num_systems, num_systems).
        '''
        scores = [
            self.score_sentence(
                hyps, references
            ) for hyps in hypotheses
        ]  # (num_systems, num_sentences)
        return self.make_pairwise_scores(scores)
    
    def rank_systems(
        self,
        hypotheses: list[list[str]],
        references: list[list[str]],
        aggregation='default'
    ):
        '''Compute ranking score for multiple systems.

        Args:
            hypotheses (list[list[str]]): Corrected sentences.
                The shape is (num_systems, num_sentences).
            references (list[list[str]]): Reference sentences.
                The shape is (num_references, num_sentences).
            aggregation: (str): How to aggregate sentence-level scores.
                - "default" follows an original aggregation, e.g., average or accumulation.
                - "trueskill" convert sentence-level scores into pairwise comparison results,
                    then apply TrueSkill. This is motivated by https://arxiv.org/abs/2502.09416.
        
        Retunrns:
            list[float]: System-level scores.
        '''
        if aggregation == "default":
            scores = [
                self.score_corpus(
                    hyps, references
                ) for hyps in hypotheses
            ]  # (num_systems, num_sentences)
        elif aggregation == 'trueskill':
            pairwise_scores = self.score_pairwise(
                hypotheses, references
            )
            scores = self.run_trueskill(pairwise_scores)
        return scores
    
def inputs_handler(
    metric: MetricBase,
    sources: list[str],
    hypotheses: list[str],
    references: list[list[str]]
) -> dict:
    '''This handles different input interface.

    Given sources, hypotheses, references,
        this function chooses the appropriate inputs
        according to the metric class.

    Returns:
        dict: The dictionary contaning some of keys from "sources", "hypotheses", and "references".
            This can be input by **<variable> to metric.score_**() functions.

    .. code-block:: python

        from gec_metrics.metrics import IMPARA, inputs_handler
        metric = ERRANT()
        inputs = inputs_handler(
            metric=metric,
            sources=[],
            hypothese=[],
            references=[[]]
        )
        metric.score_corpus(**inputs)
    '''
    if isinstance(metric, MetricBaseForReferenceBased):
        return {
            "sources": sources,
            "hypotheses": hypotheses,
            "references": references
        }
    elif isinstance(metric, MetricBaseForReferenceFree):
        return {
            "sources": sources,
            "hypotheses": hypotheses,
        }
    elif isinstance(metric, MetricBaseForSourceFree):
        return {
            "hypotheses": hypotheses,
            "references": references
        }
    else:
        raise ValueError(f'The metric class is invalid.')