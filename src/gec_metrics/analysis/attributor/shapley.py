from .base import AttributorBase
import errant
from typing import Union, Optional
import math
from gecommon import apply_edits
    
class AttributorShapley(AttributorBase):
    def __init__(self, config):
        super().__init__(config)
        
    def generate(
        self,
        src: str,
        edits: Union[list[errant.edit.Edit], list[list[errant.edit.Edit]]]
    ) -> list[dict]:
        '''Generate edited sentence by applying all patterns of edits.
        
        Args:
            src (str): source sentence.
            edits (list[errant.edit.Edit]): Edit to be applied to the source.

        Returns:
            list[Dict]: Each element has two keys:
                "sentence": An edited sentence.
                "indices": Indices of edits that were applied to the source sentence.
        '''
        edited = []
        num_edits = len(edits)
        for i in range(2 ** (num_edits)):
            # Get edit ids by binary number.
            # E.g. 5 is 101, which means first and third edits are used.
            indices = tuple(j for j in range(num_edits) if (i >> j) & 1)
            to_be_applied = [edits[j] for j in indices]
            # flatten if type(edits) is list[list[errant.edit.Edit]]
            if isinstance(edits[0], list):
                to_be_applied = [ee for e in to_be_applied for ee in e]
            sent = apply_edits(src, to_be_applied)
            edited.append({
                'sentence': sent,
                'indices': indices
            })
        return edited
    
    def post_process(
        self,
        scores: list[float],
        sent_level_score: Optional[float] = None,
        indices: Optional[list[tuple]] = None
    ) -> list[float]:
        '''Caluclate Shapley values.

        Args:
            scores (list[float]): \delta M() scores.
            sent_level_score (Optional[float]): Used when normalization.
            indices (Optional[list[Tuple]]): Which edits were applied to the source.
        
        Returns:
            list[float]: Post pocessed scores.
        '''
        def shapley_weight(n, s):
            return (math.perm(s) * math.perm(n-s-1)) / math.perm(n)
        
        assert len(scores) == len(indices)
        # In Shapley-based attribution, 2^(num_edits) sentences and their scores are used.
        # So we can know the number of edits by using log2()
        num_edits = int(math.log2(len(scores)))
        attributed_scores = [0] * num_edits
        # Create hash to access score by indices
        idx2score = {
            idx: score for idx, score in zip(indices, scores)
        }
        for i in range(num_edits):
            # We will calculate i-th edit's attributed score.
            # \boldsymbol{e} \setminus e_i in Eq.2 in the paper.
            indices_wo_i = tuple(j for j in range(num_edits) if i != j)
            # Loop for each \boldsymbol{e}' in Eq.2.
            for j in range(2 ** len(indices_wo_i)):
                # if k-th bit of j is 1, k-th edit is used.
                # E.g. j == 5 (0b101) means that the first and third edits are used.
                subset = list(indices_wo_i[k] for k in range(j) if (j >> k) & 1)
                weight = shapley_weight(num_edits, len(subset))
                # \boldsymbol{e}' \cap e_i in Eq.2.
                subset_cap_i = sorted(subset + [i])
                attributed_scores[i] += weight * (
                    idx2score[tuple(subset_cap_i)] - idx2score[tuple(subset)]
                )
        return attributed_scores
    