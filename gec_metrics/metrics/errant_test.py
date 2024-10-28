from .errant import ERRANT
import pytest
import math

SRCS = [
    'This sentences contain gramamtical error .',
    'This is a sentence .',
    'This is no change .',
]
HYPS = [
    'This sentence contains a grammatical error .',
    'the sentence was corrected into completely different one .',
    'This is no change .',
]
REFS = [
    [
        'This sentence contains a gramamtical error .',
        'dummy sentence .',
        'This is no change .',
    ],
    [
        'These sentences contain grammatical errors .',
        'another dummy sentence .',
        'This is no change .',
    ]
]
cases = [
    (0.5, 0.375, [0.7894736842105263, 0.0, 1.0]),
    (2, 0.6000000000000, [0.9375, 0.0, 1.0]),
]

class TestERRANT:
    @pytest.mark.parametrize("beta,gold_corpus_score,gold_sent_score", cases)
    def test_metric(self, beta, gold_corpus_score, gold_sent_score):
        scorer = ERRANT(ERRANT.Config(beta=beta))
        corpus_score = scorer.score_corpus(
            SRCS, HYPS, REFS
        )
        assert math.isclose(corpus_score, gold_corpus_score, abs_tol=1e-9)
        sent_score = scorer.score_sentence(
            SRCS, HYPS, REFS
        )
        assert [math.isclose(s1, s2, abs_tol=1e-9) \
                for s1, s2 in zip(sent_score, gold_sent_score)]
