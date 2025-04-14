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
        sum_scores = sum(scores)
        if sum_scores == 0:
            weight = 0
        else:
            weight = sent_level_score / sum_scores
        return [weight * s for s in scores]
    