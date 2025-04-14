import abc
import errant
from typing import Optional, Union
from dataclasses import dataclass
from gec_metrics.metrics import MetricBase
from gecommon import apply_edits

class AttributorBase(abc.ABC):
    @dataclass
    class Config:
        '''Attribution configuration.
            - metric (MetricBase): Metric instance based on gec_metrics.metrics.MetricBase
            - max_num_edits (int): Ignore a hypothesis when the number of edits exceeds this value.
            - errant_language (str): Spacy language for ERRANT.
            - quiet (bool): If False some logs will be shown.
        '''
        metric: MetricBase = None
        max_num_edits: int = float('inf')
        errant_language: str = 'en'
        quiet: bool = True

    @dataclass
    class AttributionOutput:
        '''Attribution output.
            - sent_score (float): The overall impact of edits: \delta M(S, H) = M(S, H) - M(S, S).
            - src_score (float): Source score: M(S, S).
            - attribution_scores (list[float]): Attribution score for each edit.
            - edits (list[errant.edit.Edit]): Edits extracted by ERRANT.
            - src (str): Source sentence.
        '''
        sent_score: float = None
        src_score: float = None
        attribution_scores: list[float] = None
        edits: list[errant.edit.Edit] = None
        src: str = None

    def __init__(self, config: Config):
        self.config = config
        assert isinstance(self.config.metric, MetricBase)
        self.metric = config.metric
        self.errant_annotator = errant.load(config.errant_language)

    @abc.abstractmethod
    def generate(
        self,
        src: str,
        edits: list[errant.edit.Edit]
    ) -> list[dict]:
        '''Generate edited sentence.
        How the edits are applied depends on the attribution method.
        
        Args:
            src (str): source sentence.
            edits (list[errant.edit.Edit]): Edit to be applied to the source.

        Returns:
            list[Dict]: Each element has two keys:
                "sentence": An edited sentence.
                "indices": Indices of edits that affect editing according to the setting.
        '''
    
    @abc.abstractmethod
    def post_process(
        self,
        scores: list[float],
        sent_level_score: Optional[float] = None,
        indices: Optional[list[tuple]] = None
    ) -> list[float]:
        '''Post processing depending on the method.
        E.g. normalize for one-by-one method or sum up for Shapley theory.

        Args:
            scores (list[float]): \delta M() scores.
            sent_level_score (Optional[float]): Used when normalization.
            indices (Optional[list[Tuple]]): Which edits were applied to the source.
        
        Returns:
            list[float]: Post pocessed scores.
        '''

    def attribute(
        self,
        src: str,
        hyp: Optional[str] = None,
        inputs_edits: Optional[list[errant.edit.Edit]] = None
    ) -> AttributionOutput:
        '''Calculate attribution scores.
        
        Args:
            src (str): A source sentence.
            hyp (Optional[str]): An edited sentence.
            inputs_edits (Optional[list[errant.edit.Edit]]): 
                An alternative way to pass the hyp, as edit objects.

        Returns:
            AttributorOutput: Attributor scores and related information.
        '''

        if inputs_edits is not None:
            edits = inputs_edits
            if edits != [] and isinstance(edits[0], list):
                hyp = apply_edits(
                    src,
                    [ee for e in edits for ee in e]
                )
            else:
                hyp = apply_edits(
                    src,
                    edits
                )
        else:
            assert hyp is not None
            assert self.errant_annotator is not None
            edits = self.errant_annotator.annotate(
                self.errant_annotator.parse(src),
                self.errant_annotator.parse(hyp)
            )
        empty_result = self.AttributionOutput(
            sent_score=0,
            src_score=0,
            attribution_scores=[],
            edits=[],
            src=src
        )
        if len(edits) > self.config.max_num_edits:
            if not self.config.quiet:
                print('too many edits:', len(edits))
            return empty_result
        if edits == []:
            return empty_result
        
        edited_sentences = self.generate(src, edits)
        sentences = [e['sentence'] for e in edited_sentences] + [src, hyp]
        scores = self.metric.score_sentence(
            [src] * len(sentences),
            sentences
        )
        # Source's score M(S, S)
        src_score = scores[-2]
        # Corrected sentence's score (Sentence-level score) M(S, H)
        hyp_score = scores[-1]
        # \delta M() 
        sent_level_score = hyp_score - src_score
        scores = [s - src_score for s in scores[:-2]]
        
        attribution_scores = self.post_process(
            scores,
            sent_level_score,
            indices=[e['indices'] for e in edited_sentences]
        )
        
        return self.AttributionOutput(
            sent_score=sent_level_score,
            src_score=src_score,
            attribution_scores=attribution_scores,
            edits=edits,
            src=src
        )