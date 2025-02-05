from .errant import ERRANT
import errant
from .bertscore import BertScore
from dataclasses import dataclass
from .base import MetricBaseForSourceFree

class PTERRANT(ERRANT):
    @dataclass
    class Config(ERRANT.Config):
        '''Configuration of PTERRANT
        
        - weight_model_name (str): Model to compute edit-level weights.
            Currently only "bertscore" is available.
        - weight_model_config (MetricBaseForSourceFree.Config):
            The config instance of the weight model.
            If not specified, it uses the default one.

        Also, you can use the configurations of ERRANT.
        '''
        weight_model_name: str = 'bertscore'
        weight_model_config: MetricBaseForSourceFree.Config = None

    def __init__(self, config: Config = None):
        super().__init__(config)
        name2class = {
            'bertscore': BertScore
        }
        if self.config.weight_model_name not in name2class:
            raise KeyError(f'The model name {self.config.weight_model_name} is invalid. It should be in {list(name2class.keys())}.')
        weight_model_cls = name2class[self.config.weight_model_name]
        weight_model_config = self.config.weight_model_config
        if weight_model_config is None:
            # Use default config
            weight_model_config = weight_model_cls.Config()
        self.weight_model = weight_model_cls(weight_model_config)

    def calc_edit_weights(
        self,
        src: str,
        ref: str,
        edits: list[errant.edit.Edit]
    ) -> list[float]:
        '''Calculate a weight for each edit.

        Args:
            src (str): Source sentence.
            src (str): Reference sentence.
            edits (list[errant.edit.Edit]): Edits.

        Returns:
            list[float]: The weight of each edit.
        '''
        if edits == []:
            return []
        tuple_edits = [(e.o_start, e.o_end, e.c_str) for e in edits]
        # Remove duplications
        edits = [edits[i] for i, e in enumerate(tuple_edits) if e not in tuple_edits[:i]]
        num_edits = len(edits)
        sents = [self.apply_edits(src, [e]) for e in edits]
        scores1 = self.weight_model.score_sentence([src], [[ref]]) * num_edits
        scores2 = self.weight_model.score_sentence(sents, [[ref] * num_edits])
        weights = [abs(s2 - s1) for s1, s2 in zip(scores1, scores2)]
        return {
            (e.o_start, e.o_end, e.c_str): w for e, w in zip(edits, weights)
        }

    def score_base(
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
            for ref_id, ref_edits in enumerate(ref_edits_list):
                weights = self.calc_edit_weights(
                    sources[sent_id],
                    references[ref_id][sent_id],
                    hyp_edits + ref_edits
                )
                r_edits = [(e.o_start, e.o_end, e.c_str) for e in ref_edits]
                r_types = [e.type for e in ref_edits]
                this_score = dict()
                for h_edit, h_type in zip(h_edits, h_types):
                    this_score[h_type] = this_score.get(
                        h_type, self.Score(beta=self.config.beta)
                    )
                    if h_edit in r_edits:
                        this_score[h_type].tp += weights[h_edit]
                    else:
                        this_score[h_type].fp += weights[h_edit]
                for r_edit, r_type in zip(r_edits, r_types):
                    if r_edit not in h_edits:
                        this_score[r_type] = this_score.get(
                            r_type, self.Score(beta=self.config.beta)
                        )
                        this_score[r_type].fn += weights[r_edit]
                sent_scores.append(this_score)
            scores.append(sent_scores)
        return scores
