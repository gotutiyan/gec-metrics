from .base import AttributorBase
from typing import Union, Optional
import errant
from gecommon import apply_edits

class AttributorAdd(AttributorBase):
    def __init__(self, config):
        super().__init__(config)
        
    def generate(
        self,
        src: str,
        edits: Union[list[errant.edit.Edit], list[list[errant.edit.Edit]]]
    ) -> list[dict]:
        '''Generate edited sentence by applying each edit to src.
        
        Args:
            src (str): source sentence.
            edits (list[errant.edit.Edit]): Edit to be applied to the source.

        Returns:
            list[Dict]: Each element has two keys:
                "sentence": An edited sentence.
                "indices": Indices of edits that were applied to the source sentence.
        '''
        edited = []
        for i, e in enumerate(edits):
            if isinstance(edits[0], errant.edit.Edit):
                e = [e]
            sent = apply_edits(src, e)
            edited.append({
                'sentence': sent,
                'indices': (i, )
            })
        return edited
    
    def post_process(
        self,
        scores: list[float],
        sent_level_score: Optional[float] = None,
        indices: Optional[list[tuple]] = None
    ) -> list[float]:
        '''Normalize each score by the sum of the scores.

        Args:
            scores (list[float]): \delta M() scores.
            sent_level_score (Optional[float]): Used when normalization.
            indices (Optional[list[Tuple]]): Which edits were applied to the source.
        
        Returns:
            list[float]: Post pocessed scores.
        '''
        sum_scores = sum(scores)
        if sum_scores == 0:
            weight = 0
        else:
            weight = sent_level_score / sum_scores
        return [weight * s for s in scores]
    