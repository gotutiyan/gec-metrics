from .impara import IMPARA
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
    (0.0, 0.80618088444074, [0.9970312118530273, 0.9480968117713928, 0.4734146296977997]),
    (1.01, 0.0, [0.0, 0.0, 0.0]),
]

class TestIMPARA:
    @pytest.mark.parametrize("threshold,gold_corpus_score,gold_sent_score", cases)
    def test_metric(self, threshold, gold_corpus_score, gold_sent_score):
        scorer = IMPARA(IMPARA.Config(
            model_qe='gotutiyan/IMPARA-QE',
            model_se='bert-base-cased',
            threshold=threshold
        ))
        corpus_score = scorer.score_corpus(
            SRCS, HYPS
        )
        assert math.isclose(corpus_score, gold_corpus_score, abs_tol=1e-6)
        sent_score = scorer.score_sentence(
            SRCS, HYPS
        )
        assert [math.isclose(s1, s2, abs_tol=1e-6) \
                for s1, s2 in zip(sent_score, gold_sent_score)]
    