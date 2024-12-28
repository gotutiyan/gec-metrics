from .gotoscorer import GoToScorer
import pytest
import math

SRCS = [
    'This sentences contain gramamtical error .',
    'This is a sentence .',
    'This is no change .',
]
HYPS = [
    'This sentence contains a grammatical error .',
    'this was corrected into completely different one .',
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
chunk_cls = GoToScorer.Chunk
dummy = 'DUMMY:DUMMY'
GOLD_CHUNK = [
    [
        chunk_cls(0, 0, '', dummy, 1, False),
        chunk_cls(0, 1, 'This', dummy, 1, False),
        chunk_cls(1, 1, '', dummy, 1, False),
        chunk_cls(1, 2, 'sentence', 'R:NOUN:NUM', 1, True),
        chunk_cls(2, 2, '', dummy, 1, False),
        chunk_cls(2, 3, 'contains', 'R:VERB:SVA', 1, True),
        chunk_cls(3, 3, 'a', 'M:DET', 1, True),
        chunk_cls(3, 4, 'grammatical', 'R:SPELL', 1, True),
        chunk_cls(4, 4, '', dummy, 1, False),
        chunk_cls(4, 5, 'error', dummy, 1, False),
        chunk_cls(5, 5, '', dummy, 1, False),
        chunk_cls(5, 6, '.', dummy, 1, False),
        chunk_cls(6, 6, '', dummy, 1, False),
    ],
    [
        chunk_cls(0, 0, '', dummy, 1, False),
        chunk_cls(0, 1, 'this', 'R:ORTH', 1, True),
        chunk_cls(1, 1, '', dummy, 1, False),
        chunk_cls(1, 2, 'was corrected', 'R:VERB', 1, True),
        chunk_cls(2, 2, 'into completely', 'M:OTHER', 1, True),
        chunk_cls(2, 3, 'different', 'R:OTHER', 1, True),
        chunk_cls(3, 3, '', dummy, 1, False),
        chunk_cls(3, 4, 'one', 'R:NOUN', 1, True),
        chunk_cls(4, 4, '', dummy, 1, False),
        chunk_cls(4, 5, '.', dummy, 1, False),
        chunk_cls(5, 5, '', dummy, 1, False),
    ],
    [
        chunk_cls(0, 0, '', dummy, 1, False),
        chunk_cls(0, 1, 'This', dummy, 1, False),
        chunk_cls(1, 1, '', dummy, 1, False),
        chunk_cls(1, 2, 'is', dummy, 1, False),
        chunk_cls(2, 2, '', dummy, 1, False),
        chunk_cls(2, 3, 'no', dummy, 1, False),
        chunk_cls(3, 3, '', dummy, 1, False),
        chunk_cls(3, 4, 'change', dummy, 1, False),
        chunk_cls(4, 4, '', dummy, 1, False),
        chunk_cls(4, 5, '.', dummy, 1, False),
        chunk_cls(5, 5, '', dummy, 1, False),
    ]
]
cases = [
    (0.5, 0.40540540540540543, [0.7894736842105263, 0.0, 1.0]),
    (2, 0.5357142857142857, [0.9375, 0.0, 1.0]),
]

class TestGoToScorer:
    @pytest.mark.parametrize("beta,gold_corpus_score,gold_sent_score", cases)
    def test_metric(self, beta, gold_corpus_score, gold_sent_score):
        scorer = GoToScorer(GoToScorer.Config(no_weight=True, beta=beta))
        corpus_score = scorer.score_corpus(
            SRCS, HYPS, REFS
        )
        print(corpus_score)
        assert math.isclose(corpus_score, gold_corpus_score, abs_tol=1e-9)
        sent_score = scorer.score_sentence(
            SRCS, HYPS, REFS
        )
        assert [math.isclose(s1, s2, abs_tol=1e-9) \
                for s1, s2 in zip(sent_score, gold_sent_score)]
        
    def test_chunk(self):
        scorer = GoToScorer(GoToScorer.Config(no_weight=True))
        for sent_id in range(len(SRCS)):
            edits = scorer.edit_extraction(SRCS[sent_id], HYPS[sent_id])
            chunks = scorer.generate_chunks(edits, tokens=SRCS[sent_id].split(' '))
            for c, cc in zip(chunks, GOLD_CHUNK[sent_id]):
                assert c == cc
