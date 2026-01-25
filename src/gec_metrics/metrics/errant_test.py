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

    def test_score_corpus_etype(self):
        metric = ERRANT(ERRANT.Config())
        score = metric.score_corpus_etype(SRCS, HYPS, REFS, cat=3)
        score_cls = ERRANT.Score
        assert score['R:NOUN:NUM'].__dict__ == score_cls(tp=1.).__dict__
        assert score['R:VERB:SVA'].__dict__ == score_cls(tp=1.).__dict__
        assert score['M:DET'].__dict__ == score_cls(tp=1.).__dict__
        assert score['R:SPELL'].__dict__ == score_cls(fp=1.).__dict__
        assert score['R:OTHER'].__dict__ == score_cls(fp=2., fn=1.).__dict__
        assert score['R:VERB'].__dict__ == score_cls(fp=1.).__dict__
        assert score['M:OTHER'].__dict__ == score_cls(fp=1.).__dict__
        assert score['R:NOUN'].__dict__ == score_cls(fp=1.).__dict__
