from .base import AttributorBase
import errant
from typing import Union, Optional
import itertools
import random

class AttributorShapleySampling(AttributorBase):
    def __init__(self, config):
        super().__init__(config)

    def generate(
        self,
        src: str,
        edits: list[errant.edit.Edit]
    ) -> list[dict]:
        '''Generate edited sentence by applying sampled patterns of edits.
        
        Args:
            src (str): source sentence.
            edits (list[errant.edit.Edit]): Edit to be applied to the source.

        Returns:
            list[Dict]: Each element has two keys:
                "sentence": An edited sentence.
                "indices": Indices of edits that were applied to the source sentence.
        '''
        edited = []
        indices = list(range(len(edits)))
        naive_num_samples = 1
        need_sampling = False
        for i in range(1, len(edits) + 1):
            naive_num_samples *= i
            if naive_num_samples > self.config.num_samples:
                need_sampling = True
                break

        if not need_sampling:
            orders = list(itertools.permutations(indices))
        else:
            orders = []
            used = dict()
            for _ in range(self.config.num_samples):
                while True:
                    random.shuffle(indices)
                    key = ' '.join(list(map(str, indices)))
                    if key not in used:
                        used[key] = 1
                        orders.append(indices)
                        break
            assert len(orders) == self.config.num_samples
        for order in orders:
            current_edits = []
            for i, idx in enumerate(order):
                current_edits.append(edits[idx])
                corrected = self.apply_edits(
                    src,
                    current_edits
                )
                edited.append({
                    'sentence': corrected,
                    'indices': tuple(order[:i+1])
                })
        return edited
    
    def post_process(
        self,
        scores: list[float],
        sent_level_score: Optional[float] = None,
        indices: Optional[list[tuple]] = None
    ) -> list[float]:
        '''Calculate Shapley sampling values.

        Args:
            scores (list[float]): \delta M() scores.
            sent_level_score (Optional[float]): Used when normalization.
            indices (Optional[list[Tuple]]): Which edits were applied to the source.
        
        Returns:
            list[float]: Post pocessed scores.
        '''
        num_edits = max(len(i) for i in indices)
        num_orders = len(scores) // num_edits
        assert len(scores) == num_edits * num_orders
        attributed_scores = [0] * num_edits
        for sample_id in range(num_orders):
            batch_scores = scores[sample_id * num_edits: (1+sample_id) * num_edits]
            batch_indices = indices[sample_id * num_edits: (1+sample_id) * num_edits]
            for i, edit_id in enumerate(batch_indices[-1]):
                if i == 0:
                    attributed_scores[edit_id] += batch_scores[i]
                else:
                    attributed_scores[edit_id] += \
                        batch_scores[i] - batch_scores[i - 1]
        attributed_scores = [s / num_orders for s in attributed_scores]
        return attributed_scores
