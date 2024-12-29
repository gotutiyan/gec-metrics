from .pt_errant import PTERRANT
from .bertscore import BertScore
import math
import pytest

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
    ('p', 0.42694591228288, [0.7409972173188052, 0.0, 1.0]),
    ('r', 0.25234117365929937, [0.5047669005793416, 0.0, 1.0]),
    ('f', 0.37409150009511694, [0.6618046861465671, 0.0, 1.0]),
]

class TestPTERRANT:
    @pytest.mark.parametrize("score_type,gold_corpus_score,gold_sent_score", cases)
    def test_metric(self, score_type, gold_corpus_score, gold_sent_score):
        scorer = PTERRANT(PTERRANT.Config(
            beta=0.5,
            language="en",
            weight_model_name='bertscore',
            weight_model_config=BertScore.Config(
                score_type=score_type
            )
        ))
        corpus_score = scorer.score_corpus(
            SRCS, HYPS, REFS
        )
        assert math.isclose(corpus_score, gold_corpus_score, abs_tol=1e-6)
        sent_score = scorer.score_sentence(
            SRCS, HYPS, REFS
        )
        print(sent_score)
        assert [math.isclose(s1, s2, abs_tol=1e-6) \
                for s1, s2 in zip(sent_score, gold_sent_score)]
