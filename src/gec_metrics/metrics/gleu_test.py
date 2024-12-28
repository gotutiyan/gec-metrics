from .gleu import GLEU, GLEUOfficial
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
        'This sentence contains a grammatical error .',
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
    (0.2913773302697463, [0.5, 0.0, 1.0, 0.2578313459119911]),
]
cases_official = [
    (0.369131, [0.5, 0.0, 1.0, 0.2640485]),
]

class TestGLEU:
    @pytest.mark.parametrize("gold_corpus_score,gold_sent_score", cases)
    def test_metric(self, gold_corpus_score, gold_sent_score):
        scorer = GLEU(GLEU.Config())
        corpus_score = scorer.score_corpus(
            SRCS, HYPS, REFS
        )
        assert math.isclose(corpus_score, gold_corpus_score, abs_tol=1e-9)
        sent_score = scorer.score_sentence(
            SRCS, HYPS, REFS
        )
        assert [math.isclose(s1, s2, abs_tol=1e-9) \
                for s1, s2 in zip(sent_score, gold_sent_score)]
        
    @pytest.mark.parametrize("gold_corpus_score,gold_sent_score", cases_official)
    def test_off_metric(self, gold_corpus_score, gold_sent_score):
        scorer = GLEUOfficial(GLEUOfficial.Config())
        corpus_score = scorer.score_corpus(
            SRCS, HYPS, REFS
        )
        assert math.isclose(corpus_score, gold_corpus_score, abs_tol=1e-5)
        sent_score = scorer.score_sentence(
            SRCS, HYPS, REFS
        )
        assert [math.isclose(s1, s2, abs_tol=1e-5) \
                for s1, s2 in zip(sent_score, gold_sent_score)]
    
    
    