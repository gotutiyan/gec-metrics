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
        scores = [sent_level_score - s for s in scores]
        sum_scores = sum(scores)
        if sum_scores == 0:
            weight = 0
        else:
            weight = sent_level_score / sum_scores
        return [weight * s for s in scores]
    