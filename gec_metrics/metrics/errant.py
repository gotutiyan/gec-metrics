from dataclasses import dataclass
from .base import MetricBaseForReferenceBased, MetricBase
import hashlib
import errant

class ERRANT(MetricBaseForReferenceBased):
    @dataclass
    class Config(MetricBaseForReferenceBased.Config):
        beta: float = 0.5
        language: str = 'en'

    def __init__(self, config: Config):
        super().__init__(config)
        self.errant = errant.load(config.language)
        self.cache_paarse = dict()
        self.cache_annotate = dict()

    def cached_parse(self, sent: str):
        '''Efficient parse() by caching'''
        key = hashlib.sha256(sent.encode()).hexdigest()
        if self.cache_paarse.get(key) is None:
            self.cache_paarse[key] = self.errant.parse(sent)
        return self.cache_paarse[key]
    
    def edit_extraction(self, src: str, trg: str):
        '''Extract edits given a source and a corrected.'''
        key = hashlib.sha256((src + '|||' + trg).encode()).hexdigest()
        if self.cache_annotate.get(key) is None:
            self.cache_annotate[key] = self.errant.annotate(
                self.cached_parse(src),
                self.cached_parse(trg)
            )
        return self.filter_edits(self.cache_annotate[key])
    
    def filter_edits(self, edits):
        return [e for e in edits if e.type not in ['noop', 'UNK']]
    
    def aggregate_to_overall(self, scores: dict[str, "Score"]) -> "Score":
        '''Convert error type based scores into an overall score.'''
        overall = self.Score(beta=self.config.beta)
        for v in scores.values():
            overall += v
        return overall

    def score_corpus(
        self,
        sources: list[str],
        hypotheses: list[str],
        references: list[list[str]]
    ) -> float:
        '''Calculate a corpus level score by aggregating verbose scores.'''
        verbose_scores = self.verbose_score_sentence(
            sources,
            hypotheses,
            references
        )
        score = self.Score(beta=self.config.beta)
        for v_scores in verbose_scores:  # sentence loop
            best_score = None
            for v_score_for_ref in v_scores:  # reference loop
                agg_score = self.aggregate_to_overall(v_score_for_ref)
                # The comparison is performed by adding 
                #   the current sentence-level score to the current accumulated score.
                # This is not mentioned ERRANT paper but the official implementation is doing so.
                if best_score is None or (score + best_score) < (score + agg_score):
                    best_score = agg_score
            score += best_score
        return score.f
        
    def score_sentence(
        self,
        sources: list[str],
        hypotheses: list[str],
        references: list[list[str]]
    ) -> list[float]:
        '''Calculate sentence level scores by aggregating verbose scores.
        '''
        verbose_scores = self.verbose_score_sentence(
            sources,
            hypotheses,
            references
        )
        scores = []
        for v_scores in verbose_scores:  # sentence loop
            best_score = None
            for v_score_for_ref in v_scores:  # reference loop
                agg_score = self.aggregate_to_overall(v_score_for_ref)
                if best_score is None or best_score < agg_score:
                    best_score = agg_score
            scores.append(best_score.f)
        return scores
    
    def verbose_score_sentence(
        self,
        sources: list[str],
        hypotheses: list[str],
        references: list[list[str]]
    ) -> list[dict[str, "Score"]]:
        '''Calculate scores keeping references, TP, FP, FN and error types information.
            The results can be aggregated for the purpose,
                e.g., sentence-level or corpus-level or error-type-level.
        '''
        num_sents = len(sources)
        num_refs = len(references)
        scores = []  # shape will be: (num_sents, num_refs, )
        for sent_id in range(num_sents):
            hyp_edits = self.edit_extraction(
                sources[sent_id],
                hypotheses[sent_id]
            )
            ref_edits_list = [self.edit_extraction(
                sources[sent_id],
                references[ref_id][sent_id]
            ) for ref_id in range(num_refs)]
            
            sent_scores = []  # shape will be: (num_refs, )
            h_edits = [(e.o_start, e.o_end, e.c_str) for e in hyp_edits]
            h_types = [e.type for e in hyp_edits]
            for ref_edits in ref_edits_list:
                r_edits = [(e.o_start, e.o_end, e.c_str) for e in ref_edits]
                r_types = [e.type for e in ref_edits]
                this_score = dict()
                for h_edit, h_type in zip(h_edits, h_types):
                    this_score[h_type] = this_score.get(
                        h_type, self.Score(beta=self.config.beta)
                    )
                    if h_edit in r_edits:
                        this_score[h_type].tp += 1
                    else:
                        this_score[h_type].fp += 1
                for r_edit, r_type in zip(r_edits, r_types):
                    if r_edit not in h_edits:
                        this_score[r_type] = this_score.get(
                            r_type, self.Score(beta=self.config.beta)
                        )
                        this_score[r_type].fn += 1
                sent_scores.append(this_score)
            scores.append(sent_scores)
        return scores
