from typing import Dict
from transformers import Trainer
import torch
from torch.nn import functional as F


class SkipDecodingTrainer(Trainer):
    def __init__(self, *args, skip_weight=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_weight = skip_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        
        # Enable skip decoding
        outputs = model(**inputs, skip_decoding=True)
        
        logits = outputs.logits
        skip_decisions = outputs.skip_decisions

        # Compute classification loss
        if self.label_smoother is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            loss = F.cross_entropy(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        # Compute skip decision loss
        num_layers = skip_decisions.size(1)
        target_skip_rate = torch.linspace(0, 0.5, num_layers, device=skip_decisions.device)
        actual_skip_rate = skip_decisions.mean(dim=0)
        skip_loss = F.mse_loss(actual_skip_rate, target_skip_rate)

        # Combine losses
        total_loss = loss + self.skip_weight * skip_loss

        if return_outputs:
            return total_loss, outputs
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