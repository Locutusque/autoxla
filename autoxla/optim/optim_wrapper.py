import torch_xla.core.xla_model as xm

class XLAOptimizerWrapper:
    def __init__(self, optimizer):
        self.optimizer = optimizer
    def step(*args, **kwargs):
        xm.optimizer_step(self.optimizer)