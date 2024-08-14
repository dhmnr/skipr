from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F

from transformers import Trainer
from transformers.trainer_pt_utils import (
   
    nested_detach,

)
from transformers.utils import (
   
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_mlu_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_xla_available,
    logging,
    strtobool,
)

import math



class SkipDecodingTrainer(Trainer):
    def __init__(self, *args, skip_weight=0.003, sample_K=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_weight = skip_weight
        self.sample_K = sample_K
    
    def create_optimizer(self):
        if self.optimizer is None:
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if "skip_policy" in n and p.requires_grad],
                    "weight_decay": self.args.weight_decay,
                }
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def _freeze_all_params(self):
        """Freeze all parameters except the policy network"""
        for name, param in self.model.named_parameters():
            if "skip_policy" not in name:
                param.requires_grad = False
            else:
                # print("Parameter :", name)
                param.requires_grad = True

        # Verify that at least some parameters require gradients
        if not any(p.requires_grad for p in self.model.parameters()):
            raise ValueError("No parameters have requires_grad=True. Check your model architecture.")

    def train(self, resume_from_checkpoint=None, **kwargs):
        """
        Freeze parameters before training
        """
        self._freeze_all_params()
        return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)


    def compute_loss(self, model, inputs: Dict[str, torch.Tensor], return_outputs: bool = False):
        # Replicate batch for exploration
        # Replicate batch for exploration
        batch_size = inputs['input_ids'].size(0)
        # expanded_inputs = {
        #     k: v.repeat(self.sample_K, 1) if isinstance(v, torch.Tensor) and v.dim() > 1 else v.repeat(self.sample_K)
        #     for k, v in inputs.items()
        # }

        # Forward pass
        outputs = model(**inputs, skip_decoding=True)
        # outputs2 = model(**inputs, skip_decoding=True)

        # print(outputs['logits'].shape, outputs2['logits'].shape)
        # print("Tensors are " + "Equal" if torch.equal(outputs['logits'], outputs2['logits']) else "Not Equal" )
        # raise
        # Unpack outputs
        if isinstance(outputs, tuple):
            classification_loss, logits, skip_layers, skip_probs = outputs[:4]
        elif isinstance(outputs, dict):
            classification_loss = outputs['loss']
            logits = outputs['logits']
            skip_layers = outputs['skip_layers']
            skip_probs = outputs['skip_probs']
            # print(f"loss shape {classification_loss.shape}")
            # print(f"logits shape {logits.shape}")
            # print(f"skip_layers shape {skip_layers.shape}")
            # print(f"skip_probs shape {skip_probs.shape}")
        
        else:
            raise ValueError("Unexpected output format from model")
        
        rewards = -classification_loss + self.skip_weight * skip_layers

        baseline = rewards.mean()
        advantage = rewards - baseline
        # advantage = advantage[:logits.size(0)]
        # Compute log probabilities of taken actions
        # num_layers = skip_probs.size(1)
        # layer_range = torch.arange(num_layers, device=skip_probs.device)
        # log_probs = torch.log(
        #     torch.where(layer_range[None, :] < exit_layers[:, None],
        #                 1 - skip_probs,  # Probability of not skipping
        #                 skip_probs)  # Probability of skipping at exit layer
        # )
        # print("log_probs shape", log_probs.shape)
        # log_probs = log_probs.sum(dim=1)  # Sum log probs across layers
        # print("log_probs shape", log_probs.shape)
        log_probs = skip_probs

        policy_loss = -(advantage * log_probs).mean()
        
        total_loss = policy_loss 

        if return_outputs:
            return total_loss, outputs
        # print(f'Loss: {total_loss}')
        return total_loss

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = getattr(self.model, "last_output", None)
        if output is not None and hasattr(output, "skip_decisions"):
            avg_skip_rate = output.skip_decisions.float().mean().item()
            logs["avg_skip_rate"] = avg_skip_rate

        super().log(logs)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]
        return (loss, logits, labels)

# TODO : Sample_K, loss computation 
