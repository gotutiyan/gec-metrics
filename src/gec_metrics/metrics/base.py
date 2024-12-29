import abc
from dataclasses import dataclass
import errant

class MetricBase(abc.ABC):
    @dataclass
    class Config: ...

    def __init__(self, config: Config):
        self.config = config

    def apply_edits(
        self,
        src: str,
        edits: list[errant.edit.Edit]
    ) -> str:
        '''Edit the source by the edits.
        
        Args:
            src (str): An input source sentence.
            edits (list[Edit]): An edit sequence.
        
        Returns:
            str: An edited sentence.
        '''
        # Firstly sort edits by start index.
        edits = sorted(edits, key=lambda x:x.o_start)
        offset = 0
        tokens = src.split(' ')
        for e in edits:
            if e.o_start == -1:
                continue
            s_idx = e.o_start + offset
            e_idx = e.o_end + offset
            # Is deletion edit
            if e.c_str == '':
                tokens[s_idx:e_idx] = ['$DELETE']
                offset -= (e.o_end - e.o_start) - 1
            # Is insertion edit
            elif e.o_start == e.o_end:
                tokens[s_idx:e_idx] = e.c_str.split(' ')
                offset += len(e.c_str.split())
            # Otherwise replacement edit
            else:
                tokens[s_idx:e_idx] = e.c_str.split(' ')
                offset += len(e.c_str.split(' ')) - (e.o_end - e.o_start)
        trg = ' '.join(tokens).replace(' $DELETE', '').replace('$DELETE ', '')
        return trg

class MetricBaseForReferenceBased(MetricBase, metaclass=abc.ABCMeta):
    @dataclass
    class Config(MetricBase.Config): ...

    class Score:
        def __init__(
            self,
            tp: float=0.0,
            fp: float=0.0,
            fn: float=0.0,
            tn: float=0.0,
            beta: float=0.5
        ):
            self.tp: float = tp
            self.fp: float = fp
            self.fn: float = fn
            self.tn: float = tn
            self.beta: float = beta

        def __add__(self, other) -> "Score":
            '''This overloads + operation.'''
            return self.__class__(
                tp=self.tp + other.tp,
                fp=self.fp + other.fp,
                fn=self.fn + other.fn,
                tn=self.tn + other.tn,
                beta=self.beta
            )
        
        def __lt__(self, other):
            '''This overloads < operation.
                We first compare F-score, then compare tp, then fp, finally fn.
            '''
            return [self.f, self.tp, -self.fp, -self.fn] \
                < [other.f, other.tp, -other.fp, -other.fn]

        @property
        def precision(self) -> float:
            '''This is a property so you can refer via .precision '''
            if self.fp == 0:
                return 1.0
            return self.tp / (self.tp + self.fp)
        
        @property
        def recall(self) -> float:
            '''This is a property so you can refer via .recall '''
            if self.fn == 0:
                return 1.0
            return self.tp / (self.tp + self.fn)

        @property
        def f(self) -> float:
            '''This is a property so you can refer via .f '''
            p = self.precision
            r = self.recall
            beta = self.beta
            f = float((1+(beta**2))*p*r)/(((beta**2)*p)+r) if p+r else 0.0
            return f
        
        @property
        def accuracy(self) -> float:

            if self.tp + self.fp + self.fn + self.tn == 0:
                return 0
            else:
                return (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn) 
        
        def __repr__(self):
            return f"F-{self.beta}={self.f}\n Prec={self.precision}\n Rec={self.recall}\n TP={self.tp}, FP={self.fp}, FN={self.fn}, TN={self.tn}\n"

    def score_corpus(
        self,
        sources: list[str],
        hypotheses: list[str],
        references: list[list[str]]
    ) -> float:
        '''Calculate a corpus-level score.
        By default, we use the average of sentence-level scores.

        Args:
            sources (list[str]): Source sentence.
            hypothesis (list[str]): Corrected sentences.
            references (list[list[str]]): Reference sentences.
                The shape is (the number of references, the number of sentences).
        
        Returns:
            float: The corpus-level score.
        '''
        scores = self.score_sentence(
            sources=sources,
            hypotheses=hypotheses,
            references=references
        )
        return sum(scores) / len(scores)
        
        
    @abc.abstractmethod
    def score_sentence(
        self,
        sources: list[str],
        hypotheses: list[str],
        references: list[list[str]]
    ) -> list[float]:
        '''Calculate a sentence-level score.

        Args:
            sources (list[str]): Source sentence.
            hypothesis (list[str]): Corrected sentences.
            references (list[list[str]]): Reference sentences.
                The shape is (the number of references, the number of sentences).
        
        Returns:
            list[float]: The sentence-level scores.
        '''
        raise NotImplementedError
    
    
class MetricBaseForReferenceFree(MetricBase, metaclass=abc.ABCMeta):
    
    @dataclass
    class Config(MetricBase.Config): ...

    def score_corpus(
        self,
        sources: list[str],
        hypotheses: list[str]
    ) -> float:
        '''Calculate a corpus-level score.
        By default, we use the average of sentence-level scores.

        Args:
            sources (list[str]): Source sentence.
            hypothesis (list[str]): Corrected sentences.
            references (list[list[str]]): Reference sentences.
                The shape is (the number of references, the number of sentences).
        
        Returns:
            float: The corpus-level score.
        '''
        scores = self.score_sentence(
            sources=sources,
            hypotheses=hypotheses
        )
        return sum(scores) / len(scores)
        
        
    @abc.abstractmethod
    def score_sentence(
        self,
        sources: list[str],
        hypotheses: list[str]
    ) -> list[float]:
        '''Calculate a sentence-level score.

        Args:
            sources (list[str]): Source sentence.
            hypothesis (list[str]): Corrected sentences.
            references (list[list[str]]): Reference sentences.
                The shape is (the number of references, the number of sentences).
        
        Returns:
            list[float]: The sentence-level scores.
        '''
        raise NotImplementedError
    
class MetricBaseForSourceFree(MetricBase, metaclass=abc.ABCMeta):
    '''Metric without source sentence.
        This is basically for BERTScore or BARTScore 
            (that will be a component of PT-{ERRANT, M2}.).
    '''
    @dataclass
    class Config(MetricBase.Config): ...

    def score_corpus(
        self,
        hypotheses: list[str],
        references: list[list[str]]
    ) -> float:
        '''Calculate a corpus-level score.
        By default, we use the average of sentence-level scores.

        Args:
            sources (list[str]): Source sentence.
            hypothesis (list[str]): Corrected sentences.
            references (list[list[str]]): Reference sentences.
                The shape is (the number of references, the number of sentences).
        
        Returns:
            float: The corpus-level score.
        '''
        scores = self.score_sentence(
            hypotheses=hypotheses,
            references=references
        )
        return sum(scores) / len(scores)
        
        
    @abc.abstractmethod
    def score_sentence(
        self,
        hypotheses: list[str],
        references: list[list[str]]
    ) -> list[float]:
        '''Calculate a sentence-level score.

        Args:
            sources (list[str]): Source sentence.
            hypothesis (list[str]): Corrected sentences.
            references (list[list[str]]): Reference sentences.
                The shape is (the number of references, the number of sentences).
        
        Returns:
            list[float]: The sentence-level scores.
        '''
        raise NotImplementedError