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
    
    def aggregate_score(self, scores: list["Score"]):
        '''Aggregate n-gram scores to overall score by the geometric mean.
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
        verbose_scores = self.verbose_score_sentence(
            sources,
            hypotheses,
            references
        )
        n = len(verbose_scores[0])
        overall_score = [self.Score(beta=self.config.beta) for _ in range(n)]
        # Accumulate corpus-level TP, FP, and FN for each n-gram
        for v_scores in verbose_scores:
            for i in range(n):
                overall_score[i] += v_scores[i]
        return self.aggregate_score(overall_score)

    def score_sentence(
        self,
        sources: list[str],
        hypotheses: list[str],
        references: list[list[str]]
    ) -> float:
        verbose_scores = self.verbose_score_sentence(
            sources,
            hypotheses,
            references
        )
        scores = [self.aggregate_score(v_score) for v_score in verbose_scores]
        return scores
        
    def verbose_score_sentence(
        self,
        sources: list[str],
        hypotheses: list[str],
        references: list[list[str]]
    ) -> float:
        num_sents = len(sources)
        scores = []
        for sent_id in range(num_sents):
            ngram_s = self.cached_get_all_ngrams(sources[sent_id].strip())
            ngram_h = self.cached_get_all_ngrams(hypotheses[sent_id].strip())
            ngram_rs = [
                self.cached_get_all_ngrams(ref[sent_id].strip()) for ref in references
            ]
            best_score = None
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
                if best_score is None \
                    or self.aggregate_score(best_score) < self.aggregate_score(this_score):
                    best_score = this_score
            scores.append(best_score)
        return scores