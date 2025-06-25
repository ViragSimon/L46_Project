from flwr.common import NDArrays, Scalar
import sys
from ...model import *
from flwr.client import Client, ClientApp, NumPyClient
import copy
import numpy as np
from typing import Callable, Dict, Optional, Tuple
from collections import OrderedDict
from ...dataset import load_datasets
from flwr.common import Context , ndarrays_to_parameters, ArrayRecord,  array_from_numpy,Array
from ...helper import get_parameters_size
import torch.nn as nn
from src.FedPart.FedAvg.client import FedPartAvgClient

class FedPartAdamClient(FedPartAvgClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        params_zero = [np.zeros_like(p) for p in get_parameters(self.model)]
        if "u_vectors" not in self.client_state.parameters_records:
            self.client_state.parameters_records["u_vectors"] = ArrayRecord(params_zero, keep_input=False)
        if "v_vectors" not in self.client_state.parameters_records:
             self.client_state.parameters_records["v_vectors"] = ArrayRecord(params_zero, keep_input=False)

    def _load_optimizer_state(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        u_record = self.client_state.parameters_records["u_vectors"]
        v_record = self.client_state.parameters_records["v_vectors"]
        
        # Get all stored u and v vectors
        stored_u = [torch.from_numpy(u.numpy()).to(DEVICE) for u in u_record.values()]
        stored_v = [torch.from_numpy(v.numpy()).to(DEVICE) for v in v_record.values()]
        
        # Get current model parameters
        current_params = list(self.model.parameters())
        
        # Create u and v vectors for all parameters
        initial_u = []
        initial_v = []
        
        # Map stored vectors to current parameters
        for i, param in enumerate(current_params):
            if i < len(stored_u) and i < len(stored_v):
                # Check if shapes match
                if stored_u[i].shape == param.data.shape and stored_v[i].shape == param.data.shape:
                    initial_u.append(stored_u[i])
                    initial_v.append(stored_v[i])
                else:
                    # Shape mismatch - create zero tensors with correct shape
                    print(f"[DEBUG] Shape mismatch for param {i}: stored_u shape {stored_u[i].shape}, param shape {param.data.shape}")
                    initial_u.append(torch.zeros_like(param.data))
                    initial_v.append(torch.zeros_like(param.data))
            else:
                # Fallback: create zero tensors
                initial_u.append(torch.zeros_like(param.data))
                initial_v.append(torch.zeros_like(param.data))
        
        return initial_u, initial_v

    def _save_optimizer_state(self, u: List[torch.Tensor], v: List[torch.Tensor]):
        # Convert all u and v tensors to numpy arrays
        u_arrays = [p.cpu().numpy() for p in u]
        v_arrays = [p.cpu().numpy() for p in v]
        
        self.client_state.parameters_records["u_vectors"] = ArrayRecord(u_arrays, keep_input=False)
        self.client_state.parameters_records["v_vectors"] = ArrayRecord(v_arrays, keep_input=False)
    
    def _patch_optimizer_state(self, patch: List[np.ndarray], layer_idx: int):
        if layer_idx == -1: 
            # Full optimizer state update - replace all u and v vectors
            current_u, current_v = self._load_optimizer_state()
            num_params = len(current_u)
            
            # Split patch into u and v vectors
            u_vectors = patch[0:num_params]
            v_vectors = patch[num_params:]
            
            # Update all u and v vectors
            for i in range(len(current_u)):
                if i < len(u_vectors) and i < len(v_vectors):
                    # Check if shapes match before updating
                    u_tensor = torch.from_numpy(u_vectors[i]).to(DEVICE)
                    v_tensor = torch.from_numpy(v_vectors[i]).to(DEVICE)
                    
                    if u_tensor.shape == current_u[i].shape and v_tensor.shape == current_v[i].shape:
                        current_u[i].data = u_tensor
                        current_v[i].data = v_tensor
                    else:
                        print(f"[DEBUG] Shape mismatch during patch for param {i}: u shape {u_tensor.shape}, current_u shape {current_u[i].shape}")
                        # Skip this parameter if shapes don't match
            
            self._save_optimizer_state(current_u, current_v)
        else:
            # Partial optimizer state update - only update specific layer
            current_u, current_v = self._load_optimizer_state()
            
            # For partial updates, patch should contain [u_weight, u_bias, v_weight, v_bias] for the specific layer
            if len(patch) >= 4:
                layer_param_idx = layer_idx * 2
                if layer_param_idx < len(current_u) and layer_param_idx + 1 < len(current_u):
                    # Check shapes before updating
                    u_weight = torch.from_numpy(patch[0]).to(DEVICE)
                    u_bias = torch.from_numpy(patch[1]).to(DEVICE)
                    v_weight = torch.from_numpy(patch[2]).to(DEVICE)
                    v_bias = torch.from_numpy(patch[3]).to(DEVICE)
                    
                    if (u_weight.shape == current_u[layer_param_idx].shape and 
                        u_bias.shape == current_u[layer_param_idx + 1].shape and
                        v_weight.shape == current_v[layer_param_idx].shape and
                        v_bias.shape == current_v[layer_param_idx + 1].shape):
                        
                        current_u[layer_param_idx].data = u_weight
                        current_u[layer_param_idx + 1].data = u_bias
                        current_v[layer_param_idx].data = v_weight
                        current_v[layer_param_idx + 1].data = v_bias
                        
                        self._save_optimizer_state(current_u, current_v)
                    else:
                        print(f"[DEBUG] Shape mismatch during partial patch for layer {layer_idx}")

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:

        # For LocalAdam, parameters should contain: [model_params, u_vectors, v_vectors]
        # where each section has the same length as the model parameters
        current_params = get_parameters(self.model)
        
        num_weight_params = len(current_params)
        model_patch = parameters[0:num_weight_params]
        optimizer_patch = parameters[num_weight_params:]
        
        print(f"[DEBUG] Model patch length: {len(model_patch)}")
        
        self._load_model_state()
        recieved_parameter_size = get_parameters_size(ndarrays_to_parameters(parameters))
        updated_layer_idx = int(config["updated_layers"])
        
        # Apply model updates
        if updated_layer_idx == -1:
            # Full model update
            set_parameters(self.model, model_patch, -1)
        else:
            # Partial model update - only update specific layer
            set_parameters(self.model, model_patch, updated_layer_idx)
            
        # Apply optimizer state updates
        self._patch_optimizer_state(optimizer_patch, updated_layer_idx)
    
        initial_u, initial_v = self._load_optimizer_state()
        
        trainable_layer_idx = int(config["trainable_layers"])
        freeze_layers(self.model, trainable_layer_idx)
        
        _, final_u, final_v = local_adam_train(
            model=self.model,
            train_loader=self.train_loader,
            num_epochs=self.num_epochs,
            initial_u=initial_u,
            initial_v=initial_v
        )
        
        self._save_model_state()
        self._save_optimizer_state(final_u, final_v)

        # For LocalAdam, we always return the full model parameters plus optimizer state
        # The server will handle partial updates based on the training configuration
        trained_params = get_parameters(self.model)
        u_vectors = [p.cpu().numpy() for p in final_u]
        v_vectors = [p.cpu().numpy() for p in final_v]
        
        return_ndarrays = trained_params + u_vectors + v_vectors
        metrics = {"trained_layer": trainable_layer_idx, "num_weight_params": len(trained_params), "recieved_parameter_size": recieved_parameter_size}
        
        return return_ndarrays, len(self.train_loader.dataset), metrics
    
    def get_parameters(self, config):
        """Override get_parameters to return full model parameters plus optimizer state for LocalAdam."""
        print(f"[Client {self.partition_id}] get_parameters")
        parameters = get_parameters(self.model)
        self._save_model_state()
        
        # For LocalAdam, we always return the full model parameters plus optimizer state
        # The server will handle partial updates based on the training configuration
        current_u, current_v = self._load_optimizer_state()
        u_vectors = [p.cpu().numpy() for p in current_u]
        v_vectors = [p.cpu().numpy() for p in current_v]
        
        return parameters + u_vectors + v_vectors

def get_local_adam_fed_part_avg_client_fn(dataset_loader: Callable) -> Callable:
    def local_adam_fed_part_avg_client_fn(context: Context) -> Client:
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        
        # Initialize network if not in context
        if not hasattr(context, 'model'):
            context.model = Net().to(DEVICE)
        
        trainloader, valloader, _ = dataset_loader(partition_id, num_partitions)
        return FedPartAdamClient(
            partition_id=partition_id,
            model=context.model,
            train_loader=trainloader,
            val_loader=valloader,
            num_epochs=EPOCHS,
            context=context
        ).to_client()
    
    return local_adam_fed_part_avg_client_fn
    
