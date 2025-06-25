import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Callable, Dict, Optional, Tuple
from collections import OrderedDict
from ..model import test, DEVICE
from flwr.common import NDArrays, Scalar


def get_evaluate_fn(
    testloader: DataLoader,
    model: torch.nn.Module,
) -> Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]:
    
    
    def evaluate(
        server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:

        model_copy = copy.deepcopy(model)
        params_dict = zip(model_copy.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v, device=DEVICE) for k, v in params_dict})
        
        model_keys = set(model_copy.state_dict().keys())
        params_keys = set(state_dict.keys())
        if model_keys != params_keys:
            print(f"  WARNING: Key mismatch between model and parameters!")
            print(f"  Missing in params: {model_keys - params_keys}")
            print(f"  Extra in params: {params_keys - model_keys}")
        
        model_copy.load_state_dict(state_dict, strict=True)
        model_copy.to(DEVICE)
        model_copy.eval()
    
        loss, accuracy = test(model_copy, testloader)
        print(f"  Evaluation results - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            
        return loss, {"accuracy": accuracy}
    
    return evaluate