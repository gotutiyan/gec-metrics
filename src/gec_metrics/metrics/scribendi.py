from .base import MetricBase, MetricBaseForReferenceFree
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from fuzzywuzzy.fuzz import token_sort_ratio
import abc

class Scribendi(MetricBaseForReferenceFree):
    @dataclass
    class Config(MetricBase.Config):
        model: str = 'gpt2'
        threshold: float = 0.8
        no_cuda: bool = False
        batch_size: int = 32

    def __init__(self, config: Config = None):
        super().__init__(config)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if not self.config.no_cuda:
            self.model.cuda()
    
    def score_corpus(
        self,
        sources: list[str],
        hypotheses: list[str]
    ) -> float:
        '''The corpus-level score is defined as a sum of
            sentence-level scores.

        Args:
            sources (list[str]): Source sentence.
            hypothesis (list[str]): Corrected sentences.
        
        Returns:
            float: The corpus-level scores.
        '''
        sentence_scores = self.score_sentence(
            sources,
            hypotheses
        )
        return sum(sentence_scores)
    
    def score_sentence(
        self,
        sources: list[str],
        hypotheses: list[str]
    ) -> list[float]:
        '''Calculate sentence-level scores.

        Args:
            sources (list[str]): Source sentence.
            hypothesis (list[str]): Corrected sentences.
        
        Returns:
            list[float]: The sentence-level scores.
        '''
        errorful_sources = []
        errorful_hypotheses = []
        num_sents = len(sources)
        scores = [-999] * num_sents
        original_indices = []
        for sent_id, (s, h) in enumerate(zip(sources, hypotheses)):
            if s == h:
                scores[sent_id] = 0
            else:
                errorful_sources.append(s)
                errorful_hypotheses.append(h)
                original_indices.append(sent_id)
        ppl_sources = self.ppl(errorful_sources)
        ppl_hypothesis = self.ppl(errorful_hypotheses)
        for i, (ppl_s, ppl_h) in enumerate(zip(ppl_sources, ppl_hypothesis)):
            if ppl_s <= ppl_h:
                scores[original_indices[i]] = -1
                continue
            tsr = self.token_sort_ratio(
                errorful_sources[i],
                errorful_hypotheses[i]
            )
            ldr = self.levenshtein_distance_ratio(
                errorful_sources[i],
                errorful_hypotheses[i]
            )
            if max(tsr, ldr) >= self.config.threshold:
                scores[original_indices[i]] = 1
            else:
                scores[original_indices[i]] = -1
        assert -999 not in scores  # All elements should filled in either -1, 0, 1.
        return scores
        
    def ppl(
        self,
        sents: list[str]
    ) -> list[float]:
        '''Compute perplexity using a LM.
        
        Args:
            sents (list[str]): The sentences to be computed the perplexity.

        Returns:
            list[float]: The list of perplexity.
        '''
        ppls = []
        sents = [self.tokenizer.bos_token + sent for sent in sents]
        batch_size = self.config.batch_size
        for i in range(len(sents)//batch_size+1):
            batch = sents[i*batch_size:(i+1)*batch_size]
            if len(batch) == 0:
                continue
            inputs = self.tokenizer(batch, return_tensors='pt', padding=True)
            if not self.config.no_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    labels=inputs['input_ids']
                )
                shift_logits = outputs.logits[:, :-1, :].contiguous()
                shift_labels = inputs['input_ids'][:, 1:].contiguous()
                shift_mask = inputs['attention_mask'][:, 1:].contiguous()
                batch_size, seq_len = shift_labels.shape
                loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
                loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                ).view(batch_size, seq_len)
                # The probability is normalized by the length.
                loss = (loss * shift_mask).sum(dim=1) / shift_mask.sum(dim=1)
                ppls += torch.exp(loss).tolist()
        return ppls
                
    def token_sort_ratio(self, src: str, pred: str) -> float:
        '''
        Args:
            src (str): The source sentence.
            pred (str): The corrected sentence.

        Returns:
            float: The token sort ratio.
        '''
        return token_sort_ratio(src, pred) / 100
    
    def levenshtein_distance_ratio(self, src: str, pred: str) -> float:
        '''The word-level levenshtein distance ratio.
        
        Args:
            src (str): The source sentence.
            pred (str): The corrected sentence.

        Returns:
            float: The levelshtein distance ratio.
        '''
        len_src = len(src)
        len_pred = len(pred)
        dp = [[0] * (len_pred + 1) for _ in range(len_src + 1)]
        # dp = np.zeros((len_src+1, len_pred+1))
        for i in range(1, len_src + 1):
            dp[i][0] = i
        for j in range(1, len_pred + 1):
            dp[0][j] = j
        for i in range(1, len_src + 1):
            for j in range(1, len_pred + 1):
                cost = 0
                if src[i-1] != pred[j-1]:
                    # Replacement cost is 2
                    cost = 2
                dp[i][j] = min(
                    dp[i-1][j-1] + cost,
                    min(dp[i-1][j] + 1, dp[i][j-1] + 1)
                )
        return 1 - dp[len_src][len_pred] / (len_src + len_pred)