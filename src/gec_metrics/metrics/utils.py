from transformers import (
    AutoModelForSequenceClassification,
    RobertaForSequenceClassification,
    ElectraForSequenceClassification,
    XLMRobertaForSequenceClassification
)
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import Optional, Union, Tuple

class AutoModelForSequenceClassificationMeanPool(nn.Module):
    '''An extended version of BERTForSequenceClassification to use mean pooling.
    It is currently intended for use in IMPARA.

    .. code-block:: python
        from gec_metrics.metrics.utils import AutoModelForSequenceClassificationMeanPool
        from transformers import AutoModelForSequenceClassification

        # Case1: Initizalize by giving an instance of AutoModelForSequenceClassification.
        base_model = AutoModelForSequenceClassification.from_pretrained(
            'google-bert/bert-base-uncased'
        )
        model = AutoModelForSequenceClassificationMeanPool(base_model)

        # Case 2: Use from_pretrained().
        model = AutoModelForSequenceClassificationMeanPool.from_pretrained(
            'google-bert/bert-base-uncased'
        )
    '''
    def __init__(
        self,
        model: AutoModelForSequenceClassification
    ):
        super().__init__()
        self.model = model
        self.config = model.config
        self.num_labels = model.config.num_labels

    def mean_pooling(self, hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        return torch.sum(hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        # Use a mean pool instead of CLS representation.
        pooled_output = self.mean_pooling(
            outputs.hidden_states[-1],
            attention_mask
        )
        if hasattr(self.model, 'dropout'):
            pooled_output = self.model.dropout(pooled_output)
            
        # Some implementations of the classification layer \
        #   take as input a hidden representation of all tokens \
        #   and extract only the first token in [:, 0, :].
        # So change shape from (batch, hidden) -> (batch, 1, hidden) by unsqueeze(). 
        for model_cls in [RobertaForSequenceClassification, ElectraForSequenceClassification, XLMRobertaForSequenceClassification]:
            if isinstance(self.model, model_cls):
                pooled_output = torch.unsqueeze(pooled_output, 1)
        # The classifier name is different depending on models.
        logits = None
        if hasattr(self.model, 'classifier'):
            logits = self.model.classifier(pooled_output)
        elif hasattr(self.model, 'logits_proj'):
            logits = self.model.logits_proj(pooled_output)
        assert logits is not None

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    @classmethod
    def from_pretrained(cls, name_or_path, **kwards):
        return cls(AutoModelForSequenceClassification.from_pretrained(name_or_path, **kwards))
    
    def save_pretrained(self, save_path, **kwards):
        self.model.save_pretrained(save_path, **kwards)

    @property
    def device(self):
        return self.model.device