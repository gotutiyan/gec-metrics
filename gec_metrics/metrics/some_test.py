from .some import SOME
import pytest
import math

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
    (0.55, 0.43, 0.02, 0.9251012007395426, [0.9550549093882243, 0.8325608174006145, 0.9876878754297893]),
    (0.0, 1.0, 0.0, 0.9381082322862412, [0.9672131538391113, 0.8521210352579752, 0.9949905077616373]),
]

class TestSOME:
    @pytest.mark.parametrize("weight_g,weight_f,weight_m,gold_corpus_score,gold_sent_score", cases)
    def test_metric(self, weight_g, weight_f, weight_m, gold_corpus_score, gold_sent_score):
        scorer = SOME(SOME.Config(
            weight_g=weight_g,
            weight_f=weight_f,
            weight_m=weight_m
        ))
        corpus_score = scorer.score_corpus(
            SRCS, HYPS
        )
        assert math.isclose(corpus_score, gold_corpus_score, abs_tol=1e-9)
        sent_score = scorer.score_sentence(
            SRCS, HYPS
        )
        assert [math.isclose(s1, s2, abs_tol=1e-9) \
                for s1, s2 in zip(sent_score, gold_sent_score)]
    