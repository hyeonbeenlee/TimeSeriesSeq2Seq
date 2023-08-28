import torch.nn as nn
import platform
import torch
import numpy as np


class Skeleton(nn.Module):
    model_info = {}

    def count_params(self):
        num_params = 0
        for param in self.parameters():
            num_params += np.prod(param.data.shape)
        print(f"Number of trainable parameters: {num_params:,}")
        return num_params

    def print_model_info(self):
        assert "model_info" in self.__dict__.keys()
        for k, v in self.model_info.items():
            print(f"{k.upper()}: {v}")
        print()

    def update_attributes(self, locals):  # put locals() to record configurations
        try:
            del locals["self"]  # Cannot be pickled
            del locals["__class__"]
        except KeyError:
            pass
        self.__dict__.update(locals)
        self.model_info.update(locals)
        self.model_info = {key: value for key, value in sorted(self.model_info.items())}

    def update_init_args(self, locals):
        try:
            del locals["self"]  # Cannot be pickled
        except KeyError:
            pass
        try:
            del locals["__class__"]  # Cannot be pickled
        except KeyError:
            pass
        self.model_init_args = {}
        self.model_init_args.update(locals)

    def set_device(self):
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
