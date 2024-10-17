from dataclasses import dataclass
from .base import MetricBaseForReferenceBased, MetricBase
import hashlib
import errant
import spacy

class ERRANT(MetricBaseForReferenceBased):
    @dataclass
    class Config(MetricBaseForReferenceBased.Config):
        beta: float = 0.5
        language: str = 'en'

    def __init__(self, config: Config):
        super().__init__(config)
        self.errant = errant.load(config.language)
        self.cache_parse = dict()
        self.cache_annotate = dict()

    def cached_parse(self, sent: str) -> spacy.tokens.doc.Doc:
        '''Efficient parse() by caching.
        
        Args:
            sent (str): The sentence to be parsed.
        Return:
            spacy.tokens.doc.Doc: The parse results. 
        '''
        key = hashlib.sha256(sent.encode()).hexdigest()
        if self.cache_parse.get(key) is None:
            self.cache_parse[key] = self.errant.parse(sent)
        return self.cache_parse[key]
    
    def edit_extraction(
        self, src: str, trg: str
    ) -> list[errant.edit.Edit]:
        '''Extract edits given a source and a corrected.

        Args:
            src (str): The source sentence.
            trg (str): The corrected sentence.
        
        Returns:
            list[errant.edit.Edit]: Extracted edits.
        '''
        key = hashlib.sha256((src + '|||' + trg).encode()).hexdigest()
        if self.cache_annotate.get(key) is None:
            self.cache_annotate[key] = self.errant.annotate(
                self.cached_parse(src),
                self.cached_parse(trg)
            )
        return self.filter_edits(self.cache_annotate[key])
    
    def filter_edits(
        self,
        edits: list[errant.edit.Edit]
    ) -> list[errant.edit.Edit]:
        '''Handle edits that will be ignored.'''
        return [e for e in edits if e.type not in ['noop', 'UNK']]
    
    def aggregate_to_overall(self, scores: dict[str, "Score"]) -> "Score":
        '''Convert error type-wise scores into an overall score.
        
        Args:
            scores (dict[str, "Score"]): Error type-wise scores.
        
        Returns:
            Score: The aggregated score.
        '''
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
        '''Calculate a corpus level score by aggregating verbose scores.

        Args:
            sources (list[str]): Source sentence.
            hypothesis (list[str]): Corrected sentences.
            references (list[list[str]]): Reference sentences.
                The shape is (the number of references, the number of sentences).
        
        Returns:
            float: The corpus-level F-beta score.
        '''
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

        Args:
            sources (list[str]): Source sentence.
            hypothesis (list[str]): Corrected sentences.
            references (list[list[str]]): Reference sentences.
                The shape is (the number of references, the number of sentences).
        
        Returns:
            list[float]: The sentence-level F-beta scores.
        '''
        verbose_scores = self.verbose_score_sentence(
            sources,
            hypotheses,
            references
        )
        scores = []
        for sent_id, v_scores in enumerate(verbose_scores):  # sentence loop
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
    ) -> list[list[dict[str, "Score"]]]:
        '''Calculate scores while retaining sentence and reference boundaries.
            The results can be aggregated according to the purpose,
                e.g., at sentence-level or corpus-level.

        Args:
            sources (list[str]): Source sentence.
            hypothesis (list[str]): Corrected sentences.
            references (list[list[str]]): Reference sentences.
                The shape is (the number of references, the number of sentences).
        
        Returns:
            list[list[dict[str, "Score"]]]: The verbose scores.
                - The list shape is (num_sents, num_refs)
                - The dict contains error type-wise scores.
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
