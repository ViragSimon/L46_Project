from flwr.common import NDArrays, Scalar
import sys
from ...model import *
from flwr.client import Client, ClientApp, NumPyClient
import copy
import numpy as np
import torch
from typing import Callable, Dict, Optional, Tuple
from collections import OrderedDict
from ...dataset import load_datasets
from flwr.common import Context , ndarrays_to_parameters,   ParametersRecord,  array_from_numpy,Array
from ...model import set_parameters, get_parameters, fedAvg_train, test, DEVICE, EPOCHS
from ...helper import get_parameters_size
import torch.nn as nn
from ...LocalAdam.FedAvg.client import FedPartAdamClient

class FedPartProxAdamClient(FedPartAdamClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

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

        global_params = copy.deepcopy(self.model).parameters()
            
        # Apply optimizer state updates
        self._patch_optimizer_state(optimizer_patch, updated_layer_idx)
    
        initial_u, initial_v = self._load_optimizer_state()
        
        trainable_layer_idx = int(config["trainable_layers"])
        freeze_layers(self.model, trainable_layer_idx)
        
        _, final_u, final_v = proximal_local_adam_train(
            model=self.model,
            train_loader=self.train_loader,
            num_epochs=self.num_epochs,
            initial_u=initial_u,
            initial_v=initial_v,
            proximal_mu=config["proximal_mu"],
            global_params=global_params
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

def get_local_adam_fed_part_prox_client_fn(dataset_loader: Callable) -> Callable:
    def local_adam_fed_part_prox_client_fn(context: Context) -> Client:
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        if not hasattr(context, 'model'):
            context.model = Net().to(DEVICE)
        
        train_loader, val_loader, _ = dataset_loader(partition_id, num_partitions)
        
        return FedPartProxAdamClient(
            partition_id=partition_id,
            model=context.model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=EPOCHS,
            context=context
        ).to_client()
    return local_adam_fed_part_prox_client_fn
        
        
        

        
