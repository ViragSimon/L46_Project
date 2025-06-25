from ...model import *
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, ndarrays_to_parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays, FitIns, FitRes, EvaluateIns, EvaluateRes, NDArrays
from typing import Optional, List, Tuple, Union
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.client_proxy import ClientProxy
from ...helper import get_parameters_size, serialize_optimizer_state, deserialize_optimizer_state
from ...FedPart.FedAvg.strategy import FedPartAvg
import sys
import torch
import copy
from functools import partial, reduce
import numpy as np
from src.Fedmom2.FedAvg.strategy import FedAvgMom2

"""
Implementation of this file based on the previous project implementation but has been greatly modified and extneded
"""


class FedProxMom2(FedAvgMom2):
    def __init__(self, proximal_mu:float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_optim_state = None
        self.proximal_mu = proximal_mu
        

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        
        config = {"trainable_layers": self.layer_training_sequence[self.training_sequence_index], "updated_layers": self.updated_layers, "proximal_mu": self.proximal_mu}

        if self.global_optim_state is not None:
            config["optimizer_state"] = self.global_optim_state
        
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        print(f"Training on layer {self.layer_training_sequence}")
        fit_configurations = []

        params_array = parameters_to_ndarrays(parameters)
        
        # If doing full model update, send all parameters
        if self.layer_training_sequence[self.training_sequence_index] == -1 or self.updated_layers == -1:
            selected_params = parameters
        else:
            layer_idx = self.updated_layers
            selected_params = ndarrays_to_parameters([
                    params_array[layer_idx * 2],     # Weight
                    params_array[layer_idx * 2 + 1]  # Bias
                ])

        for idx, client in enumerate(clients):
            fit_configurations.append((client, FitIns(selected_params, config)))

        self.updated_layers = self.layer_training_sequence[self.training_sequence_index]
        self.training_sequence_index = self.training_sequence_index + 1
        
        return fit_configurations
        

    
        
        
        
        