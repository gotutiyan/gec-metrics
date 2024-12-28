from .green import GREEN
import pytest
import math

SRCS = [
    'This sentences contain gramamtical error .',
    'This is a sentence .',
    'This is no change .',
    'I bought a pen to write a letter to a friend .'
]
HYPS = [
    'This sentence contains a grammatical error .',
    'the sentence was corrected into completely different one .',
    'This is no change .',
    'I bought a pen to write a letter to a friend .'
]
REFS = [
    [
        'This sentence contains a gramamtical error .',
        'dummy sentence .',
        'This is no change .',
        'I bought a pen to write a letter to the friend .'
    ],
    [
        'These sentences contain grammatical errors .',
        'another dummy sentence .',
        'This is no change .',
        'I bought a pen to write the letter to the friend .'
    ]
]
cases = [
    (0.5, 67.353281938837, [67.080694940041, 34.099683242425, 100.0, 89.496113939418]),
    (2.0, 71.852523831581, [74.467815149281, 60.992075408949, 100.0, 68.051848160668]),
]

class TestGREEN:
    @pytest.mark.parametrize("beta,gold_corpus_score,gold_sent_score", cases)
    def test_metric(self, beta, gold_corpus_score, gold_sent_score):
        scorer = GREEN(GREEN.Config(beta=beta))
        corpus_score = scorer.score_corpus(
            SRCS, HYPS, REFS
        )
        assert math.isclose(100*corpus_score, gold_corpus_score, abs_tol=1e-9)
        sent_score = scorer.score_sentence(
            SRCS, HYPS, REFS
        )
        assert [math.isclose(100*s1, s2, abs_tol=1e-9) \
                for s1, s2 in zip(sent_score, gold_sent_score)]
