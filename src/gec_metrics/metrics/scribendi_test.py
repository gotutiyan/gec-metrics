from .scribendi import Scribendi
import pytest

SRCS = [
    'This sentences contain gramamtical error .',
    'This is a sentence .',
    'This is no change .'
]
HYPS = [
    'This sentence contains a grammatical error .',
    'the sentence was corrected into completely different one .',
    'This is no change .'
]
cases = [
    ('gpt2', 0.8, 0, [1, -1, 0]),
    ('gpt2', 1.0, -2, [-1, -1, 0])
]

class TestScribendi:
    @pytest.mark.parametrize("model,threshold,gold_corpus_score,gold_sent_score", cases)
    def test_metric(self, model, threshold, gold_corpus_score, gold_sent_score):
        scorer = Scribendi(Scribendi.Config(
            model=model,
            threshold=threshold
        ))
        corpus_score = scorer.score_corpus(
            SRCS, HYPS
        )
        assert corpus_score == gold_corpus_score
        sent_score = scorer.score_sentence(
            SRCS, HYPS
        )
        assert sent_score == gold_sent_score
    