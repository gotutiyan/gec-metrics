from .utils import AutoModelForSequenceClassificationMeanPool
import torch

class TestMeanPool:
    def test(self):
        model = AutoModelForSequenceClassificationMeanPool.from_pretrained(
            'google-bert/bert-base-uncased'
        )
        # assume (num_sents, num_tokens, hidden_size) = (1, 6, 1)
        hidden_state = torch.tensor([[
            [1],[2],[3],[4],[5],[6]
        ]])
        # assume (num_sents, num_tokens) = (1, 6)
        attention_mask = torch.tensor([[
            1,1,1,1,0,0
        ]])
        mean_pool = model.mean_pooling(hidden_state, attention_mask)
        assert torch.isclose(
            mean_pool,
            torch.tensor([[2.5]])
        )
