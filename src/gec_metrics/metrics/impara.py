from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizer
)
from .base import MetricBase, MetricBaseForReferenceFree
from .utils import AutoModelForSequenceClassificationMeanPool
import torch
import torch.nn as nn
import math
from dataclasses import dataclass

class SimilarityEstimator(MetricBaseForReferenceFree):
    @dataclass
    class Config(MetricBaseForReferenceFree.Config):
        '''Similarity Estimator configuration.

        Args:
            model (str): Model name to compute similarity.
            batch_size (int): Batch size during inference.
            max_length (int): Maximum length in tokenization. 
                The input is truncated if longer than it.
            no_cuda (bool): If True, it will work on CPU.
        '''
        model: str = 'google-bert/bert-base-cased'
        batch_size: int = 32
        max_length: int = 128
        no_cuda: bool = False

    def __init__(self, config: Config=None):
        super().__init__(config)
        self.model = AutoModel.from_pretrained(self.config.model).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        if not self.config.no_cuda and torch.cuda.is_available():
            self.model.cuda()

    def score_sentence(
        self,
        sources: list[str],
        hypotheses: list[str]
    ) -> list[float]:
        '''Compute similarity scores.

        Args:
            sources (list[str]): Source sentence.
                The shape is (num_sentences, )
            hypotheses (list[str]): Corrected sentences.
                The shape is (num_sentences, )
        
        Returns:
            list[float]: The similarity scores.
        '''
        bsz = self.config.batch_size
        assert len(sources) == len(hypotheses)
        similarities = None
        tokenizer_args = {
            'max_length': self.config.max_length,
            'padding': "max_length",
            'truncation': True,
            'return_tensors': 'pt'
        }
        for i in range(0, len(sources), bsz):
            src_encode = self.tokenizer(sources[i: i+bsz], **tokenizer_args)
            hyp_encode = self.tokenizer(hypotheses[i: i+bsz], **tokenizer_args)
            src_encode = {k: v.to(self.model.device) for k, v in src_encode.items()}
            hyp_encode = {k: v.to(self.model.device) for k, v in hyp_encode.items()}
            out = self.forward(
                src_encode['input_ids'],
                src_encode['attention_mask'],
                hyp_encode['input_ids'],
                hyp_encode['attention_mask'],
            )
            if similarities is None:
                similarities = out.view(-1)
            else:
                similarities = torch.cat([similarities, out.view(-1)], dim=-1)
        assert len(similarities) == len(sources)
        return similarities

    @torch.no_grad()
    def forward(
        self,
        src_input_ids: torch.Tensor,
        src_attention_mask: torch.Tensor,
        pred_input_ids: torch.Tensor,
        pred_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        '''Compute the cosine similarity given source and corrected sentences.

        Args:
            src_input_ids (torch.Tensor): Tokenized source sentences.
                The shape is (num_batch, sequence_length)
            src_attention_mask (torch.Tensor): The attention mask to handle padding.
                The shape is (num_batch, sequence_length)
            pred_input_ids (torch.Tensor): Tokenized corrected sentences.
                The shape is (num_batch, sequence_length)
            pred_attention_mask (torch.Tensor): The attention mask to handle padding.
                The shape is (num_batch, sequence_length)
        
        Returns:
            torch.Tensor: The cosine similarity.
                The shape is (num_batch, )
        '''
        src_state = self.model(
            src_input_ids,
            src_attention_mask
        ).last_hidden_state
        pred_state = self.model(
            pred_input_ids,
            pred_attention_mask
        ).last_hidden_state
        src_pooler = self.mean_pooling(src_state, src_attention_mask)
        trg_pooler = self.mean_pooling(pred_state, pred_attention_mask)
        cosine_sim = nn.CosineSimilarity()
        similarity = cosine_sim(src_pooler, trg_pooler)
        return similarity

    def mean_pooling(
        self,
        states: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        '''Compute mean pooling. Only the representaion with mask==1 are used.
        
        Args:
            states (torch.Tensor): The token-level representation.
                The shape is (num_batch, sequence_length, hidden_size)
            mask: torch.Tensor: The mask indicates padding or not.
                The shape is (num_batch, sequence_length)
        
        Returns:
            torch.Tensor: The mean pooled representation.
                The shape is (num_batch, hidden_size)
        '''
        states[mask == 0] = 0 # batch x seq_len x hidden
        sum_logits = torch.sum(states, dim=1) # batch x hidden
        length = torch.sum(mask, dim=-1) # batch x
        pooled_logits = torch.div(sum_logits.transpose(1, 0), length).transpose(1, 0) # batch x hidden
        return pooled_logits
    
    @property
    def device(self):
        return self.model.device

class IMPARA(MetricBaseForReferenceFree):
    @dataclass
    class Config(MetricBase.Config):
        '''IMPARA configuration.

        Args:
            model_qe (str): Quality estimation model.
            model_se (str): Similarity estimation model.
            pooling (str): Pooling method. 'cls' or 'mean'.
            max_length (int): Maximum length of inputs.
            threshold (float): Threshold for the similarity score.
            no_cuda (bool): If True, work on CPU.
            batch_size (int): Batch size for the inference.
        '''
        model_qe: str = 'gotutiyan/IMPARA-QE'
        model_se: str = 'google-bert/bert-base-cased'
        pooling: str = 'cls'
        max_length: int = 128
        threshold: float = 0.9
        no_cuda: bool = False
        batch_size: int = 32

    def __init__(self, config: Config=None):
        super().__init__(config)
        assert self.config.pooling in ['cls', 'mean'], "The config.pooling should be in ['cls', 'mean']."
        self.similarity_estimator = SimilarityEstimator(SimilarityEstimator.Config(
            model=self.config.model_se,
            batch_size=self.config.batch_size,
            max_length=self.config.max_length,
            no_cuda=self.config.no_cuda
        ))
        self.model_qe = AutoModelForSequenceClassification.from_pretrained(self.config.model_qe).eval()
        self.tokenizer_qe = AutoTokenizer.from_pretrained(self.config.model_qe)
        if not self.config.no_cuda and torch.cuda.is_available():
            self.model_qe.cuda()
        if self.config.pooling == 'mean':
            # Wrap the model to use a mean pooling instead of CLS representation.
            self.model_qe = AutoModelForSequenceClassificationMeanPool(
                self.model_qe
            )

    def score_sentence_se(
        self, sources: list[str], hypotheses: list[str]
    ) -> list[float]:
        '''Compute similarity scores.

        Args:
            sources (list[str]): Source sentence.
                The shape is (num_sentences, )
            hypotheses (list[str]): Corrected sentences.
                The shape is (num_sentences, )
        
        Returns:
            list[float]: The similarity scores.
        '''
        return self.similarity_estimator.score_sentence(
            sources, hypotheses
        )

    def score_sentence_qe(
        self, hypotheses: list[str]
    ) -> list[float]:
        '''Compute quality scores.

        Args:
            hypotheses (list[str]): Corrected sentences.
                The shape is (num_sentences, )
        
        Returns:
            list[float]: The quality scores.
        '''
        batch_size = self.config.batch_size
        scores = None
        for i in range(0, len(hypotheses), batch_size):
            tokenizer_args = {
                'max_length': self.config.max_length,
                'padding': "max_length",
                'truncation': True,
                'return_tensors': 'pt'
            }
            batch = hypotheses[i: i+batch_size]
            hyp_encode_qe = self.tokenizer_qe(
                batch, **tokenizer_args
            )
            hyp_encode_qe = {k: v.to(self.model_qe.device) for k, v in hyp_encode_qe.items()}
            qe_scores = self.model_qe(
                hyp_encode_qe['input_ids'],
                hyp_encode_qe['attention_mask']
            ).logits.view(-1)
            qe_scores = torch.sigmoid(qe_scores)
            if scores is None:
                scores = qe_scores
            else:
                scores = torch.cat([scores, qe_scores], dim=-1)
        assert len(scores) == len(hypotheses)
        return scores
        
    @torch.no_grad()
    def score_sentence(
        self,
        sources: list[str],
        hypotheses: list[str]
    ) -> list[float]:
        '''Calculate sentence-level scores.

        Args:
            sources (list[str]): Source sentence.
                The shape is (num_sentences, )
            hypotheses (list[str]): Corrected sentences.
                The shape is (num_sentences, )
        
        Returns:
            list[float]: The sentence-level scores.
        '''
        scores = []
        assert len(sources) == len(hypotheses)
        se_scores = self.score_sentence_se(
            sources, hypotheses
        )
        qe_scores = self.score_sentence_qe(
            hypotheses
        )
        qe_scores[se_scores <= self.config.threshold] = 0
        scores += qe_scores.tolist()
        assert len(scores) == len(sources)
        return scores