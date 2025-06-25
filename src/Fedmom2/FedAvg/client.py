
from flwr.common import NDArrays, Scalar
import sys
from ...model import *
from flwr.client import Client, ClientApp, NumPyClient
import copy
import numpy as np
from typing import Callable, Dict, Optional, Tuple
from collections import OrderedDict
from ...dataset import load_datasets
from flwr.common import Context , ndarrays_to_parameters,   ParametersRecord,  array_from_numpy,Array
from ...model import set_parameters, get_parameters, fedAvg_train, test, DEVICE, EPOCHS
from ...helper import get_parameters_size, serialize_optimizer_state, deserialize_optimizer_state
from src.FedPart.FedAvg.client import FedPartAvgClient

class FedAvgMom2Client(FedPartAvgClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    
    def _get_optimizer_state(self):
        state_dict = self.optimizer.state_dict()
        return state_dict
    
    def _set_optimizer_state(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def fit(self, parameters, config):
        print(f"[Client {self.partition_id}] fit")

        self._load_model_state()
        received_parameter_size = get_parameters_size(ndarrays_to_parameters(parameters))
        set_parameters(self.model, parameters, config["updated_layers"])
        freeze_layers(self.model, config["trainable_layers"])

        self.optimizer = torch.optim.Adam(self.model.parameters())
        if "optimizer_state" in config:
            print(f"Got optimizer state in config, setting..")
            serialized_optimizer_state = config["optimizer_state"]
            optim_state = deserialize_optimizer_state(serialized_optimizer_state)
            self._set_optimizer_state(optim_state)

        fedAvgMom_train(self.model, self.train_loader, num_epochs=EPOCHS, optimizer=self.optimizer)

        # handle optimizer state (serialize before passing)
        optim_state = self._get_optimizer_state()
        serialized_optimizer_state = serialize_optimizer_state(optim_state)
        print(f"After training, got optim state {serialized_optimizer_state[:50]}")

        new_config =  {
            "trained_layer":config["trainable_layers"], 
            "recieved_parameter_size": received_parameter_size,
            "optimizer_state": serialized_optimizer_state
            }

        self._save_model_state()

        return self.get_parameters(config), len(self.train_loader), new_config




def get_fedavg_mom2_client_fn(dataset_loader: Callable) -> Callable:
    def fedavg_mom2_client_fn(context: Context) -> Client:
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        

        if not hasattr(context, 'model'):
            context.model = Net().to(DEVICE)
        
        train_loader, val_loader, _ = dataset_loader(partition_id, num_partitions)
        
        return FedAvgMom2Client(
            partition_id=partition_id,
            model=context.model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=EPOCHS,
            context=context
        ).to_client()
    return fedavg_mom2_client_fn