from .base import AttributorBase
import errant
from typing import Union, Optional
from gecommon import apply_edits

class AttributorSub(AttributorBase):
    def __init__(self, config):
        super().__init__(config)
        
    def generate(
        self,
        src: str,
        edits: Union[list[errant.edit.Edit], list[list[errant.edit.Edit]]]
    ) -> list[dict]:
        '''Generate edited sentence by removing each edit from the reference.
        
        Args:
            src (str): source sentence.
            edits (list[errant.edit.Edit]): Edit to be applied to the source.

        Returns:
            list[Dict]: Each element has two keys:
                "sentence": An edited sentence.
                "indices": Indices of edits that were removed from the reference sentence.
        '''
        edited = []
        for i, e in enumerate(edits):
            # Get edits without i-th edit
            to_be_applied = [e for j, e in enumerate(edits) if j != i]
            # flatten if type(edits) is list[list[errant.edit.Edit]]
            if isinstance(edits[0], list):
                to_be_applied = [ee for e in to_be_applied for ee in e]
            sent = apply_edits(src, to_be_applied)
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
        scores = [sent_level_score - s for s in scores]
        sum_scores = sum(scores)
        if sum_scores == 0:
            weight = 0
        else:
            weight = sent_level_score / sum_scores
        return [weight * s for s in scores]
    