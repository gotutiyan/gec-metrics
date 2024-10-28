import math
from collections import Counter
from typing import List, Tuple
import numpy as np
import random
from .base import MetricBaseForReferenceBased
from dataclasses import dataclass
from tqdm import tqdm
import time
from .green import GREEN

class GLEU(GREEN):
    '''GLEU using GREEN reformulation (https://aclanthology.org/2024.inlg-main.25.pdf).
    '''
    @dataclass
    class Config(MetricBaseForReferenceBased.Config):
        '''
        Args:
            iter (int): The number of iterations.
            n (int): The maximum n of ngram.
            unit (str): Word-level or character-level. Can be 'word' or 'char'.
        '''
        iter: int = 500
        n: int = 4
        unit: str = 'word'
    
    def aggregate_score(
        self,
        scores: list["Score"],
        hyp_len: int,
        ref_len: int
    ) -> float:
        '''Aggregate n-gram scores to an overall score by the geometric mean.
        
        Args:
            scores (list[Score]): The scores keeping n-gram boundary.
                The shape is (n, )
            hyp_len (int): The length of the hypothesis.
            ref_len (int): The length of the reference.
        
        Returns:
            float: The aggregated score.
        '''
        log_bp = min(0, 1 - ref_len / hyp_len)
        ps = [s.precision for s in scores]
        
        if any(p <= 0 for p in ps):
            return 0
        else:
            log_prec = sum(math.log(p) for p in ps) / self.config.n
        return math.exp(log_bp + log_prec)
    
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
        verbose_scores, hyp_lens, ref_lens = self.score_base(
            sources,
            hypotheses,
            references
        )
        scores = []  # The shape will be (num_iters,)
        for iter_id, iter_scores in enumerate(verbose_scores):
            ngram_wise_scores = [self.Score() for _ in range(self.config.n)]
            corpus_level_hyp_len = 0
            corpus_level_ref_len = 0
            for sent_id, sent_score in enumerate(iter_scores):
                for n, ngram_score in enumerate(sent_score):
                    ngram_wise_scores[n] += ngram_score
                corpus_level_hyp_len += hyp_lens[iter_id][sent_id]
                corpus_level_ref_len += ref_lens[iter_id][sent_id]
            s = self.aggregate_score(
                ngram_wise_scores,
                corpus_level_hyp_len,
                corpus_level_ref_len
            )
            scores.append(s)
        # Average of iterations
        score = sum(scores) / len(scores)
        return score

    def score_sentence(
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
            list[float]: The sentence-level scores.
        '''
        # The number of iteration is always 1
        #   because we do not need to draw a reference.
        original_iter = self.config.iter
        self.config.iter = 1
        scores = [[] for _ in range(len(sources))]  # The shape will be (num_sents, num_iters=1)
        for i in range(len(references)):
            verbose_scores, hyp_lens, ref_lens = self.score_base(
                sources,
                hypotheses,
                references[i:i+1]  # Use only i-th reference
            )
            for iter_id, iter_scores in enumerate(verbose_scores):
                for sent_id, sent_score in enumerate(iter_scores):
                    # Aggregate ngram-wise score to an overall score
                    s = self.aggregate_score(
                        sent_score,
                        hyp_len=hyp_lens[iter_id][sent_id],
                        ref_len=ref_lens[iter_id][sent_id]
                    )
                    scores[sent_id].append(s)
        # Average of references
        scores = [sum(s) / len(s) for s in scores]
        self.config.iter = original_iter
        return scores
        
    def score_base(
        self,
        sources: list[str],
        hypotheses: list[str],
        references: list[list[str]]
    ) -> float:
        '''Compute True Positive and False Negative using GREEN's reformulation.
            (https://aclanthology.org/2024.inlg-main.25.pdf)

            The actual equation is (TI + TK - UD) / (TI + TK + OI + UD),
                thus we regard 
                - True Positive (TP) as TI + TK - UD,
                - False Positive (FP) as OI + 2*UD.
            Finally, precision = TP / (TP+FP) will be the GLEU score.

        Args:
            sources (list[str]): Source sentence.
            hypothesis (list[str]): Corrected sentences.
            references (list[list[str]]): Reference sentences.
                The shape is (the number of references, the number of sentences).
        
        Returns:
            list[list[list["Score"]]]: The verbose scores.
                The shape is (num_iterations, num_sents, max_ngram).
            list[list[int]]: The length for the hypotheses. 
                The shape is (num_iterations, num_sents)
            list[list[int]]: The length for the references. 
                The shape is (num_iterations, num_sents) 
        '''
        num_sents = len(sources)
        num_refs = len(references)
        all_ref_lens = [
            [len(r.split(' ')) for r in ref] for ref in references
        ]
        scores = []  # The shape will be (num_iters, num_sents, max_ngram)
        ref_lens = []  # (num_iters, num_sents)
        hyp_lens = [[len(h.split(' ')) for h in hypotheses]] * self.config.iter  # (num_iters, num_sents)
        cached_score = np.zeros((num_refs, num_sents)).tolist()

        for sent_id in range(num_sents):
            ngram_s = self.cached_get_all_ngrams(sources[sent_id])
            ngram_h = self.cached_get_all_ngrams(hypotheses[sent_id])
            for ref_id in range(num_refs):
                ngram_r = self.cached_get_all_ngrams(references[ref_id][sent_id])
                this_score = [self.Score() for _ in range(self.config.n)]
                for ngram in ngram_h:
                    idx = len(ngram) - 1
                    ms = ngram_s.get(ngram, 0)
                    mh = ngram_h.get(ngram, 0)
                    mr = ngram_r.get(ngram, 0)
                    ti = max(min(mr, mh) - ms, 0)
                    tk = min(ms, mh, mr)
                    oi = max(mh - max(ms, mr), 0)
                    ud = max(min(ms, mh) - mr, 0)
                    s = this_score[idx]
                    # TI
                    s.tp += ti
                    # TK
                    s.tp += tk
                    # OI
                    s.fp += oi
                    # UD
                    s.tp -= ud
                    s.fp += 2 * ud
                cached_score[ref_id][sent_id] = this_score

        for iter_id in range(self.config.iter):
            # Draw the ids of the references.
            # As same as official implementation, we fix seed `iter_id * 101`.
            random.seed(iter_id*101)
            sampled_ref_ids = [random.randint(0, num_refs - 1) for _ in range(num_sents)]
            # Extract the sampled reference and its length.
            ref_lens.append([
                all_ref_lens[ref_id][sent_id] \
                    for sent_id, ref_id in enumerate(sampled_ref_ids)
            ])
            this_iter_scores = [cached_score[ref_id][sent_id] \
                    for sent_id, ref_id in enumerate(sampled_ref_ids)]
            scores.append(this_iter_scores)
        return scores, hyp_lens, ref_lens
    
class GLEUOfficial(GLEU):
    def score_base(
        self,
        sources: list[str],
        hypotheses: list[str],
        references: list[list[str]]
    ) -> Tuple[list[list[list["Score"]]], list[list[int]], list[list[int]]]:
        '''The official implementation contains an error
                where the frequency of n-grams is ignored in the calculation of S\R. 
            As a result, when an n-gram is classified into both TK and UD, 
                it is entirely counted as TK.

        Args:
            sources (list[str]): Source sentence.
            hypothesis (list[str]): Corrected sentences.
            references (list[list[str]]): Reference sentences.
                The shape is (the number of references, the number of sentences).
        
        Returns:
            list[list[list["Score"]]]: The verbose scores.
                The shape is (num_iterations, num_sents, max_ngram).
            list[list[int]]: The length for the hypotheses. 
                The shape is (num_iterations, num_sents)
            list[list[int]]: The length for the references. 
                The shape is (num_iterations, num_sents) 
        '''
        num_sents = len(sources)
        num_refs = len(references)
        all_ref_lens = [
            [len(r.split(' ')) for r in ref] for ref in references
        ]
        scores = []  # The shape will be (num_iters, num_sents, max_ngram)
        ref_lens = []  # (num_iters, num_sents)
        # The length of hypthesis is the same in all iterations.
        hyp_lens = [[len(h.split(' ')) for h in hypotheses]] * self.config.iter  # (num_iters, num_sents)
        # The score is determined by the sentence id and the reference id,
        #   thus we pre-compute and cache them.
        # In the iterations we only sample the reference ids.
        cached_score = np.zeros((num_refs, num_sents)).tolist()

        for sent_id in range(num_sents):
            ngram_s = self.cached_get_all_ngrams(sources[sent_id])
            ngram_h = self.cached_get_all_ngrams(hypotheses[sent_id])
            for ref_id in range(num_refs):
                ngram_r = self.cached_get_all_ngrams(references[ref_id][sent_id])
                this_score = [self.Score() for _ in range(self.config.n)]
                for ngram in ngram_h:
                    idx = len(ngram) - 1
                    ms = ngram_s.get(ngram, 0)
                    mh = ngram_h.get(ngram, 0)
                    mr = ngram_r.get(ngram, 0)
                    ti = max(min(mr, mh) - ms, 0)
                    tk = min(ms, mh, mr)
                    oi = max(mh - max(ms, mr), 0)
                    ud = max(min(ms, mh) - mr, 0)
                    # If TK > 0 and UD > 0, the official implementation treats both of them as TK.
                    # Considering that the TP includes "-UD" and the FP includes "2*UD",
                    #   this can handle by reducing UD by half and push the rest onto TK.
                    # For example, when a ngram has TK=2 and UD=1,
                    #   In the correct GLEU is
                    #       TP = TK-UD = 2 - 1 = 1
                    #       FP = 2*UD = 2 * 1 = 2
                    #       Precision (GLEU) = TP / (TP+FP) = 1/3
                    #   For the official implementation,
                    #       we preprocess: TK ← TK + UD/2 = 2.5, UD ← UD/2 = 0.5
                    #       TP = TK-UD = 2.5 - 0.5 = 2
                    #       FP = 2*UD = 1
                    #       Precision (GLEU) = TP / (TP+FP) = 2/3
                    # Treating UD as TK causes the difference between 1/3 and 2/3.
                    if tk > 0:
                        ud = ud / 2
                        tk += ud
                    s = this_score[idx]
                    # TI
                    s.tp += ti
                    # TK
                    s.tp += tk
                    # OI
                    s.fp += oi
                    # UD
                    s.tp -= ud
                    s.fp += 2 * ud
                # The official implementation also takes max(TP, 0).
                # This means we add |TP| if (TK+TI-UD) < 0.
                # We handle this by adding TP to FP when TP<0, and set TP to zero.
                for i in range(self.config.n):
                    if this_score[i].tp < 0:
                        this_score[i].fp += this_score[i].tp
                        this_score[i].tp = 0
                cached_score[ref_id][sent_id] = this_score

        for iter_id in range(self.config.iter):
            # Draw the ids of the references.
            # As same as official implementation, we fix seed `iter_id * 101`.
            random.seed(iter_id*101)
            sampled_ref_ids = [random.randint(0, num_refs - 1) for _ in range(num_sents)]
            ref_lens.append([
                all_ref_lens[ref_id][sent_id] \
                    for sent_id, ref_id in enumerate(sampled_ref_ids)
            ])
            this_iter_scores = [cached_score[ref_id][sent_id] \
                                for sent_id, ref_id in enumerate(sampled_ref_ids)]
            scores.append(this_iter_scores)
        return scores, hyp_lens, ref_lens