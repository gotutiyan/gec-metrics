from .base import MetricBaseForReferenceBased
from .errant import ERRANT
import errant 
from dataclasses import dataclass
from collections import Counter
import json

class GoToScorer(ERRANT):
    @dataclass
    class Config(ERRANT.Config):
        beta: float = 0.5
        weight_file: str = ''
        ref_id: int = 0  # GoToScorer uses only one reference.
        no_weight: bool = False

    @dataclass
    class Chunk:
        o_start: int = 0
        o_end: int = 0
        c_str: str = ''
        type: str = ''
        weight: float = 1.0
        is_edited: bool = False

    def __init__(self, config: Config):
        super().__init__(config)
        if not config.no_weight:
            data = json.load(open(
                self.config.weight_file
            ))
            self.weights = [d['weights'] for d in data]

    def generate_chunks(
        self,
        edits: list[errant.edit.Edit],
        tokens: list[str]
    ) -> list[Chunk]:
        '''Generate a chunk sequence given an edit sequence.
            - Tokens included in each edit become a single chunk.
            - Each token outside of the edits beocome a chunk respectively.
            - Additionally, dummy chunks will be inserted between all tokens 
                to account for possible insertions.
        Args:
            edits (list[errant.edit.Edit]):
                The edit sequence that can be obtained via errant.annotate()
            tokens (list[str]):
                The source tokens.
        Return:
            list[Chunk]: The chunk sequence.
        '''
        chunks = []
        edit_idx = 0
        # We first convert edits into chunks.
        word_id = 0
        while word_id < len(tokens):
            if edit_idx < len(edits) \
                and edits[edit_idx].o_start == word_id:
                e = edits[edit_idx]
                chunks.append(self.Chunk(
                    o_start=e.o_start,
                    o_end=e.o_end,
                    c_str=e.c_str,
                    type=e.type,
                    is_edited=True
                ))
                edit_idx += 1
                word_id = e.o_end - 1
            else:
                chunks.append(self.Chunk(
                    o_start=word_id,
                    o_end=word_id + 1,
                    c_str=tokens[word_id],
                    type='DUMMY:DUMMY',
                    is_edited=False
                ))
            word_id += 1
        if len(edits) > 0 and edits[-1].o_start == len(tokens):
            # This handles the insertion edit at the end of the sentence.
            # e.g. SRC='This is a' and TRG='This is a sentence',
            #   The edit will be like (3, 3, 'sentence').
            chunks.append(self.Chunk(
                o_start=edits[-1].o_start,
                o_end=edits[-1].o_end,
                c_str=edits[-1].c_str,
                type=edits[-1].type,
                is_edited=True
            ))
        # We then insert dummy chunk for the potential insertion.
        is_previous_chunk_insertion = False
        new_chunks = []
        for chunk in chunks:
            # If there is already an insetion chunk.
            if chunk.o_start == chunk.o_end:
                is_previous_chunk_insertion = True
                new_chunks.append(chunk)
                continue
            else:
                # Otherweise, we insert a dummy chunk.
                if not is_previous_chunk_insertion:
                    # Insert a dummy chunk.
                    new_chunks.append(self.Chunk(
                        o_start=chunk.o_start,
                        o_end=chunk.o_start,
                        c_str='',
                        type='DUMMY:DUMMY',
                        is_edited=False
                    ))
                new_chunks.append(chunk)
                is_previous_chunk_insertion = False
        if not is_previous_chunk_insertion:
            # The last chunk to handle the insetion edit at the end.
            new_chunks.append(self.Chunk(
                o_start=chunks[-1].o_end,
                o_end=chunks[-1].o_end,
                c_str='',
                type='DUMMY:DUMMY',
                is_edited=False
            ))
        return new_chunks
        
    def verbose_score_sentence(
        self,
        sources: list[str],
        hypotheses: list[str],
        references: list[list[str]]
    ) -> list[dict[str, "Score"]]:
        '''We calculate the scores while preserving all boundaries 
            of the sentences, references, and error types.

        Args:
            source (list[str]): The source sentences.
            hypotheses (list[str]): The source sentences.
            references (list[list[str]]): The references sentences.
                The shape is (num_references, num_sentences).
        Return:
            list[dict[str, "Score"]]: The verbose scores.
                The list length is the same as the number of sentences.
                The dict format is {error type: Score()}.
        '''
        num_sents = len(sources)
        num_refs = len(references)
        assert 0 <= self.config.ref_id < num_refs
        scores = list()  # The shape will be (num_sents, )
        for sent_id in range(num_sents):
            hyp_edits = self.edit_extraction(
                sources[sent_id],
                hypotheses[sent_id]
            )
            ref_edits = self.edit_extraction(
                sources[sent_id],
                references[self.config.ref_id][sent_id]
            )
            hyp_chunks = self.generate_chunks(
                hyp_edits,
                tokens=sources[sent_id].split(' ')
            )
            ref_chunks = self.generate_chunks(
                ref_edits,
                tokens=sources[sent_id].split(' ')
            )
            no_weight = self.config.no_weight
            if not no_weight:
                weights = self.weights[sent_id]
                assert len(ref_chunks) == len(weights)
            this_score = dict()
            for i, r_chunk in enumerate(ref_chunks):
                r_chunk.weight = 1.0 if no_weight else weights[i]
                is_correct, try_edit = self.annotate(
                    r_chunk, hyp_chunks
                )
                this_score[r_chunk.type] = this_score.get(
                    r_chunk.type,
                    self.Score(beta=self.config.beta)
                )
                s = this_score[r_chunk.type]  
                if try_edit:
                    if is_correct:
                        s.tp += r_chunk.weight
                    else:
                        s.fp += r_chunk.weight
                        if r_chunk.is_edited:
                            s.fn += r_chunk.weight
                else:
                    if is_correct:
                        s.tn += r_chunk.weight
                    else:
                        s.fn += r_chunk.weight
            scores.append([this_score])
        return scores

    def annotate(
        self,
        r_chunk: Chunk,
        hyp_chunks: list[Chunk],
    ) -> tuple[bool]:
        '''Annotate whether the reference chunk is correct
            and whether the system attempted to edit it.

        Args:
            r_chunk (Chunk):
                The chunk to be evaluated.
            hyp_chunks (list[Chunk]):
                The chunk sequence for one GEC systems.
        Return
            tuple[bool]: This contains two elements.
                The first one represents correctness.
                The second one represents whether the system tried to edit or not.
        '''
        is_correct = False
        try_edit = False
        for h_chunk in hyp_chunks:
            if (r_chunk.o_start, r_chunk.o_end) \
                == (h_chunk.o_start, h_chunk.o_end):
                if r_chunk.c_str == h_chunk.c_str:
                    is_correct = True
                else:
                    is_correct = False
                # To distuinguish TP or TN.
                try_edit = h_chunk.is_edited
                break
            elif r_chunk.o_start == r_chunk.o_end:
                is_correct = True
            elif (r_chunk.o_start <= h_chunk.o_start < r_chunk.o_end) \
                or (h_chunk.o_start <= r_chunk.o_start < h_chunk.o_end):
                try_edit |= h_chunk.is_edited
        return is_correct, try_edit
    
    def visualize_chunk(
        self,
        chunks: list[Chunk],
        tokens: str
    ) -> None:
        '''The visualizer.

        Args:
            chunks (list[Chunk]):
                The chunk sequence.
            tokens (list[str]):
                The source tokens.
        ```
            from gec_metrics.metrics.gotoscorer import GoToScorer
            scorer = GoToScorer(GoToScorer.Config(no_weight=True))
            src = 'This sentences contain gramamtical error .'
            trg = 'This sentence contains a grammatical error .'
            edits = scorer.edit_extraction(src, trg)
            chunks = scorer.generate_chunks(edits, src.split(' '))
            scorer.visualize_chunk(chunks, src.split(' '))

            # Output:
            # |   |This|   |sentences|   |contain |   |gramamtical|   |error|   | . |   |
            # |   |This|   |sentence |   |contains| a |grammatical|   |error|   | . |   |
            # |1.0|1.0 |1.0|   1.0   |1.0|  1.0   |1.0|    1.0    |1.0| 1.0 |1.0|1.0|1.0|
        ```
        '''
        def insert_space(w, n):
            if len(w) < n:
                offset = n - len(w)
                w = ' '*(offset//2) + w + ' '*(offset - offset//2)
            return w
        
        vis = {
            'orig': ['orig  : '],
            'cor': ['gold  : '],
            'weight': ['weight: '],
            'type': ['cat   : ']
        }
        for c in chunks:
            orig = ' '.join(tokens[c.o_start:c.o_end])
            cor = c.c_str
            weight_str = str(round(c.weight, 2))
            etype = '' if c.type == 'DUMMY:DUMMY' else c.type
            max_len = max(len(orig), len(cor), len(weight_str), len(etype))
            vis['orig'].append(insert_space(orig, max_len))
            vis['cor'].append(insert_space(cor, max_len))
            vis['weight'].append(insert_space(weight_str, max_len))
            vis['type'].append(insert_space(etype, max_len))
        print('|'.join(vis['orig']) + '|')
        print('|'.join(vis['cor']) + '|')
        print('|'.join(vis['weight']) + '|')
        print('|'.join(vis['type']) + '|')
        print()
        