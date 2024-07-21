from typing import Dict
import torch
from torch.nn import functional as F

from transformers import Trainer
import math


class SkipDecodingTrainer(Trainer):
    def __init__(self, *args, skip_weight=0.1, sample_K=8, **kwargs):
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
            classification_loss, logits, exit_layers, skip_probs = outputs[:4]
        elif isinstance(outputs, dict):
            classification_loss = outputs['loss']
            logits = outputs['logits']
            exit_layers = outputs['exit_layers']
            skip_probs = outputs['skip_probs']
            # print(f"loss shape {classification_loss.shape}")
            # print(f"logits shape {logits.shape}")
            # print(f"exit_layers shape {exit_layers.shape}")
            # print(f"skip_probs shape {skip_probs.shape}")
        
        else:
            raise ValueError("Unexpected output format from model")

        batch_size = math.ceil(logits.size(0) / self.args.n_gpu)
        # Calculate rewards (negative loss)
        
        rewards = -classification_loss

        # Compute advantage (reward - baseline)
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
        # Compute policy loss
        policy_loss = -(advantage * log_probs).mean()
        
        # Total loss (you can adjust the entropy coefficient)
        total_loss = policy_loss 

        if return_outputs:
            return total_loss, outputs
        print(f'Loss: {total_loss}')
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


# TODO : Sample_K, loss computation 