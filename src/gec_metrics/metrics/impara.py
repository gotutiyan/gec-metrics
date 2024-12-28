from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizer
)
from .base import MetricBase, MetricBaseForReferenceFree
import torch
import torch.nn as nn
import math
from dataclasses import dataclass

class SimilarityEstimator(nn.Module):
    '''This class computes cosine similarity.
    '''
    def __init__(self, model_id: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()
    
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
            src_input_ids (torch.Tensor): Tokenized corrected sentences.
                The shape is (num_batch, sequence_length)
            src_attention_mask (torch.Tensor): The attention mask to handle padding.
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
        model_qe: str = 'gotutiyan/IMPARA-QE'
        model_se: str = 'bert-base-cased'
        max_length: int = 128
        threshold: float = 0.9
        no_cuda: bool = False
        batch_size: int = 32

    def __init__(self, config: Config):
        super().__init__(config)
        self.model_qe = AutoModelForSequenceClassification.from_pretrained(config.model_qe).eval()
        self.tokenizer_qe = AutoTokenizer.from_pretrained(config.model_qe)
        self.model_se = SimilarityEstimator(config.model_se).eval()
        self.tokenizer_se = AutoTokenizer.from_pretrained(config.model_se)
        if not config.no_cuda:
            self.model_qe.cuda()
            self.model_se.cuda()
    
    def score_sentence(
        self,
        sources: list[str],
        hypotheses: list[str]
    ) -> list[float]:
        '''Calculate sentence-level scores.

        Args:
            sources (list[str]): Source sentence.
            hypotheses (list[str]): Corrected sentences.
        
        Returns:
            list[float]: The sentence-level scores.
        '''
        scores = []
        batch_size = self.config.batch_size
        for i in range(math.ceil(len(sources) / batch_size)):
            tokenizer_args = {
                'max_length': self.config.max_length,
                'padding': "max_length",
                'truncation': True,
                'return_tensors': 'pt'
            }
            hyp_encode_qe = self.tokenizer_qe(
                hypotheses[i*batch_size:(i+1)*batch_size],
                **tokenizer_args
            )
            src_encode_se = self.tokenizer_se(
                sources[i*batch_size:(i+1)*batch_size],
                **tokenizer_args
            )
            hyp_encode_se = self.tokenizer_se(
                hypotheses[i*batch_size:(i+1)*batch_size],
                **tokenizer_args
            )
            hyp_encode_qe = {k: v.to(self.model_qe.device) for k, v in hyp_encode_qe.items()}
            src_encode_se = {k: v.to(self.model_se.device) for k, v in src_encode_se.items()}
            hyp_encode_se = {k: v.to(self.model_se.device) for k, v in hyp_encode_se.items()}
            with torch.no_grad():
                qe_scores = self.model_qe(
                    hyp_encode_qe['input_ids'],
                    hyp_encode_qe['attention_mask']
                ).logits.view(-1)
                se_scores = self.model_se(
                    src_encode_se['input_ids'],
                    src_encode_se['attention_mask'],
                    hyp_encode_se['input_ids'],
                    hyp_encode_se['attention_mask'],
                ).view(-1)
            qe_scores = torch.sigmoid(qe_scores)
            qe_scores[se_scores < self.config.threshold] = 0
            scores += qe_scores.tolist()
        return scores