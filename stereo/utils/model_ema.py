import copy
import math

import torch


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


class ModelEMA:
    """Maintain an evaluation-only exponential moving average of model state."""

    def __init__(self, model, decay=0.9999, tau=200.0, updates=0):
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        if tau < 0:
            raise ValueError("EMA tau must be non-negative")
        self.ema = copy.deepcopy(unwrap_model(model)).eval()
        self.decay_rate = float(decay)
        self.tau = float(tau)
        self.updates = int(updates)
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)

    def decay(self):
        if self.tau == 0:
            return self.decay_rate
        return self.decay_rate * (1.0 - math.exp(-self.updates / self.tau))

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = self.decay()
        source_state = unwrap_model(model).state_dict()
        for name, ema_value in self.ema.state_dict().items():
            source_value = source_state[name].detach().to(device=ema_value.device)
            if ema_value.is_floating_point():
                ema_value.mul_(decay).add_(
                    source_value.to(dtype=ema_value.dtype), alpha=1.0 - decay
                )
            else:
                ema_value.copy_(source_value)

    def load_state_dict(self, state_dict, updates=0):
        self.ema.load_state_dict(state_dict)
        self.updates = int(updates)
