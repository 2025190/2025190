import torch
import torch.nn as nn
from torchvision.ops import RoIAlign

class FeatureExtractor:
    def __init__(self):
        self.features = {}
        self.hooks = []
        
    def get_hook(self, name):
        def hook_fn(module, input, output):
            self.features[name] = output
        return hook_fn
    
    def register_hooks(self, model, layers_to_extract):
        if hasattr(model, 'model') and hasattr(model.model, 'model'):
            seq_model = model.model.model
            for name, idx in layers_to_extract.items():
                if isinstance(idx, int) and 0 <= idx < len(seq_model):
                    module = seq_model[idx]
                    hook = module.register_forward_hook(self.get_hook(name))
                    self.hooks.append(hook)
                print(f"Registered hook to layer: {name} and {idx}")
                
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_features(self):
        return self.features
    
    def __del__(self):
        self.remove_hooks() 