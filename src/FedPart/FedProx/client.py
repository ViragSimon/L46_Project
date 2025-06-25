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
from ...model import set_parameters, get_parameters, fedAvg_train, test, DEVICE, EPOCHS, fedProx_train
from ...helper import get_parameters_size
from ...FedPart.FedAvg.client import FedPartAvgClient


class FedPartProxClient(FedPartAvgClient):
    def __init__(self, partition_id, model, train_loader, val_loader, num_epochs: int, context: Context):
        super().__init__(partition_id, model, train_loader, val_loader, num_epochs, context)

    def fit(self, parameters, config):
        print(f"[Client {self.partition_id}] fit, config: {config} with parameters size {parameters}")
        self._load_model_state()
        recieved_parameter_size = get_parameters_size(ndarrays_to_parameters(parameters))     
        set_parameters(self.model, parameters, config["updated_layers"])
        global_params = copy.deepcopy(self.model).parameters()
        freeze_layers(self.model, config["trainable_layers"])
        fedProx_train(self.model, self.train_loader, EPOCHS, config["proximal_mu"], global_params)
        self._save_model_state()
        return self.get_parameters(config), len(self.train_loader), {"trained_layer":config["trainable_layers"], "recieved_parameter_size": recieved_parameter_size}

def get_fedpart_prox_client_fn(dataset_loader: Callable) -> Callable:
    def fedpart_prox_client_fn(context: Context) -> Client:
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        if not hasattr(context, 'model'):
            context.model = Net().to(DEVICE)
        
        train_loader, val_loader, _ = dataset_loader(partition_id, num_partitions)
        
        return FedPartProxClient(
            partition_id=partition_id,
            model=context.model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=EPOCHS,
            context=context
        ).to_client()
    return fedpart_prox_client_fn
        
