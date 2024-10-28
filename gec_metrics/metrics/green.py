from .base import MetricBaseForReferenceBased
from dataclasses import dataclass
from collections import Counter
import math
import hashlib

class GREEN(MetricBaseForReferenceBased):
    @dataclass
    class Config(MetricBaseForReferenceBased.Config):
        n: int = 4
        beta: float = 2.0
        unit: str = 'word'

    def __init__(self, config: Config):
        super().__init__(config)
        self.cache_ngram = dict()
    
    def cached_get_all_ngrams(
        self,
        sentence: str,
    ) -> dict[str, int]:
        '''Get frequency of n-gram for all n (min_n <= n <= max_n)
        '''
        if sentence == '':
            return dict()
        if self.config.unit == 'word':
            words = sentence.split(' ')
        elif self.config.unit == 'char':
            words = sentence
        key = hashlib.sha256(sentence.encode()).hexdigest()
        if self.cache_ngram.get(key) is None:
            ngrams = []
            for n in range(1, self.config.n + 1):
                for i in range(len(words) - n + 1):
                    ngrams.append(tuple(words[i:i+n]))
            self.cache_ngram[key] = Counter(ngrams)
        return self.cache_ngram[key]
    
    def aggregate_score(self, scores: list["Score"]) -> float:
        '''Aggregate n-gram scores to an overall score by the geometric mean.
        
        Args:
            scores (list[Score]): The scores keeping n-gram boundary.
                The shape is (n, )
        
        Returns:
            float: The aggregated score.
        '''
        ps = [s.precision for s in scores]
        rs = [s.recall for s in scores]
        if 0 in ps:
            prec = 0
        else:
            # $(\PI x)^(1/N) = exp((1/N) \sum log(x))
            prec = math.exp(sum(math.log(p) for p in ps) / len(scores))
        if 0 in rs:
            rec = 0
        else:
            rec = math.exp(sum(math.log(r) for r in rs) / len(scores))
        beta = self.config.beta
        f = float((1+(beta**2))*prec*rec) / (((beta**2)*prec)+rec) if prec+rec else 0.0
        return f
    
    def score_corpus(
        self,
        sources: list[str],
        hypotheses: list[str],
        references: list[list[str]]
    ) -> float:
        '''Calculate sentence level scores by aggregating verbose scores.

        Args:
            sources (list[str]): Source sentence.
            hypothesis (list[str]): Corrected sentences.
            references (list[list[str]]): Reference sentences.
                The shape is (the number of references, the number of sentences).
        
        Returns:
            float: The corpus-level scores.
        '''
        verbose_scores = self.score_base(
            sources,
            hypotheses,
            references
        )
        score = [self.Score(beta=self.config.beta) for _ in range(self.config.n)]
        for v_scores in verbose_scores:  # sentence loop
            best_score = None
            for v_score_for_ref in v_scores:  # reference loop
                # Choose the best reference
                if best_score is None \
                    or self.aggregate_score(best_score) < self.aggregate_score(v_score_for_ref):
                    best_score = v_score_for_ref
            # Accumulate scores for each n-gram.
            for n in range(self.config.n):
                score[n] += best_score[n]
        return self.aggregate_score(score)

    def score_sentence(
        self,
        sources: list[str],
        hypotheses: list[str],
        references: list[list[str]]
    ) -> list[float]:
        '''Calculate sentence level scores by aggregating verbose scores.

        Args:
            sources (list[str]): Source sentence.
            hypothesis (list[str]): Corrected sentences.
            references (list[list[str]]): Reference sentences.
                The shape is (the number of references, the number of sentences).
        
        Returns:
            list[float]: The sentence-level scores.
        '''
        verbose_scores = self.score_base(
            sources,
            hypotheses,
            references
        )
        scores = []
        for v_scores in verbose_scores:  # sentence loop
            best_score = None
            for v_score_for_ref in v_scores:  # reference loop
                # Choose the best reference
                if best_score is None \
                    or self.aggregate_score(best_score) < self.aggregate_score(v_score_for_ref):
                    best_score = v_score_for_ref
            # Accumulate scores for each n-gram.
            scores.append(self.aggregate_score(best_score))
        return scores
        
    def score_base(
        self,
        sources: list[str],
        hypotheses: list[str],
        references: list[list[str]]
    ) -> list[list[list["Score"]]]:
        '''Calculate scores while retaining sentence and reference boundaries.
            The results can be aggregated according to the purpose,
                e.g., at sentence-level or corpus-level.

        Args:
            sources (list[str]): Source sentence.
            hypothesis (list[str]): Corrected sentences.
            references (list[list[str]]): Reference sentences.
                The shape is (the number of references, the number of sentences).
        
        Returns:
            list[list[list["Score"]]]: The verbose scores.
                The shape is (num_iterations, num_sents, max_ngram).
        '''
        num_sents = len(sources)
        scores = []  # The shape will be (num_sents, num_refs, max_ngram)
        for sent_id in range(num_sents):
            ngram_s = self.cached_get_all_ngrams(sources[sent_id].strip())
            ngram_h = self.cached_get_all_ngrams(hypotheses[sent_id].strip())
            ngram_rs = [
                self.cached_get_all_ngrams(ref[sent_id].strip()) for ref in references
            ]
            sent_score = []
            for ngram_r in ngram_rs:
                all_ngram = set(list(ngram_s.keys()) + list(ngram_h.keys()) + list(ngram_r.keys()))
                this_score = [self.Score(beta=self.config.beta) for _ in range(self.config.n)]
                for ngram in all_ngram:
                    idx = len(ngram) - 1
                    ms = ngram_s.get(ngram, 0)
                    mh = ngram_h.get(ngram, 0)
                    mr = ngram_r.get(ngram, 0)
                    # TD
                    this_score[idx].tp += max(
                        ms - max(mr, mh),
                        0
                    )
                    # TI
                    this_score[idx].tp += max(
                        min(mr, mh) - ms,
                        0
                    )
                    # TK
                    this_score[idx].tp += min(ms, mh, mr)
                    # OD
                    this_score[idx].fp += max(
                        min(ms, mr) - mh,
                        0
                    )
                    # OI
                    this_score[idx].fp += max(
                        mh - max(ms, mr),
                        0
                    )
                    # UD
                    this_score[idx].fn += max(
                        min(ms, mh) - mr,
                        0
                    )
                    # UI
                    this_score[idx].fn += max(
                        mr - max(ms, mh),
                        0
                    )
                sent_score.append(this_score)
            scores.append(sent_score)
        return scores