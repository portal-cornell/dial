from transformers import GemmaPreTrainedModel, GemmaModel
from transformers.loss.loss_utils import LOSS_MAPPING
from transformers.modeling_outputs import ModelOutput
from peft import PeftModelForSequenceClassification, PeftType
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass

class PeftModelForDualHeads(PeftModelForSequenceClassification):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        sequence_labels=None,
        causal_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        peft_config = self.active_peft_config
        assert not peft_config.is_prompt_learning
        assert peft_config.peft_type == PeftType.LORA
        if not peft_config.is_prompt_learning:
            with self._enable_peft_forward_hooks(**kwargs):
                kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    sequence_labels=sequence_labels,
                    causal_labels=causal_labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )

@dataclass
class DualLMOutputWithPast(ModelOutput):
    sequence_loss: Optional[torch.FloatTensor] = None
    causal_loss: Optional[torch.FloatTensor] = None
    sequence_logits: torch.FloatTensor = None
    causal_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class GemmaForDualHeads(GemmaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = GemmaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.sequence_loss_function = LOSS_MAPPING["ForSequenceClassification"]
        self.causal_loss_function = LOSS_MAPPING["ForCausalLM"]

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        sequence_labels: Optional[torch.LongTensor] = None,
        causal_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_logits_to_keep: int = 0
    ) -> DualLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        assert return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        sequence_logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(sequence_logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = sequence_logits[torch.arange(batch_size, device=sequence_logits.device), sequence_lengths]
        sequence_loss = None
        if sequence_labels is not None:
            sequence_loss = self.sequence_loss_function(logits=sequence_logits, labels=sequence_labels, pooled_logits=pooled_logits, config=self.config)

        causal_logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
        causal_loss = None
        if causal_labels is not None:
            causal_loss = self.causal_loss_function(causal_logits, causal_labels, self.vocab_size)

        return DualLMOutputWithPast(
            sequence_loss=sequence_loss,
            causal_loss=causal_loss,
            sequence_logits=pooled_logits,
            causal_logits=causal_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )