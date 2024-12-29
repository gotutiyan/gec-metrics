from .bertscore import BertScore
import math
import pytest

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
# (num_refs, num_sents) -> (num_sents, num_refs)
# REFS = list(zip(*REFS))
cases = [
    ('p', 0.6699924468994141, [0.8371637463569641, 0.17281359434127808, 1.0]),
    ('r', 0.7383550703525543, [0.8531531691551208, 0.3619120419025421, 1.0]),
    ('f', 0.6951882441838583, [0.6618046861465671, 0.0, 1.0]),
]

class TestBertScore:
    @pytest.mark.parametrize("score_type,gold_corpus_score,gold_sent_score", cases)
    def test_metric(self, score_type, gold_corpus_score, gold_sent_score):
        scorer = BertScore(BertScore.Config(
            score_type=score_type,
            rescale_with_baseline=True,
        ))
        corpus_score = scorer.score_corpus(
            HYPS, REFS
        )
        assert math.isclose(corpus_score, gold_corpus_score, abs_tol=1e-6)
        sent_score = scorer.score_sentence(
            HYPS, REFS
        )
        assert [math.isclose(s1, s2, abs_tol=1e-6) \
                for s1, s2 in zip(sent_score, gold_sent_score)]
