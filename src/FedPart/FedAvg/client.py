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
from ...helper import get_parameters_size



class FedPartAvgClient(NumPyClient):
    def __init__(self, partition_id, model, train_loader, val_loader, num_epochs: int, context: Context):
        print(f"[Client {partition_id}] initialized")
        self.partition_id = partition_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.client_state = context.state
        if "net_parameters" not in self.client_state.parameters_records:
            self.client_state.parameters_records["net_parameters"] = ParametersRecord()
            self._save_model_state()

    def _save_model_state(self):
        p_record = ParametersRecord()
        parameters = get_parameters(self.model)
        
        for i, param in enumerate(parameters):
            p_record[f"layer_{i}"] = Array(param)
        
        self.client_state.parameters_records["net_parameters"] = p_record

    def _load_model_state(self):
        p_record = self.client_state.parameters_records["net_parameters"]
        parameters = []
        
        for i in range(len(p_record)):
            parameters.append(p_record[f"layer_{i}"].numpy())
        
        set_parameters(self.model, parameters)

    def get_parameters(self, config):
        print(f"[Client {self.partition_id}] get_parameters")
        parameters = get_parameters(self.model)
        trainable_layer = config["trainable_layers"]
        self._save_model_state()
        
        if trainable_layer == -1:
            return parameters
        
        trained_layer = [parameters[trainable_layer*2], parameters[trainable_layer*2 +1]]
        return trained_layer

    def fit(self, parameters, config):
        print(f"[Client {self.partition_id}] fit, config: {config}")
        
        self._load_model_state()
        recieved_parameter_size = get_parameters_size(ndarrays_to_parameters(parameters))  
        set_parameters(self.model, parameters, config["updated_layers"])
        freeze_layers(self.model, config["trainable_layers"])
        fedAvg_train(self.model, self.train_loader, num_epochs=self.num_epochs)
        
        self._save_model_state()
            
        return self.get_parameters(config), len(self.train_loader), {"trained_layer":config["trainable_layers"], "recieved_parameter_size": recieved_parameter_size}
    

    def evaluate(self, parameters, config):
        print(f"[Client {self.partition_id}] evaluate")
        self._load_model_state()
        current_state = get_parameters(self.model)
        set_parameters(self.model, current_state)
        loss, accuracy = test(self.model, self.val_loader)
        return float(loss), len(self.val_loader), {"accuracy": float(accuracy)}

def get_fedpart_avg_client_fn(dataset_loader: Callable) -> Callable:
    def fedpart_avg_client_fn(context: Context) -> Client:
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        
        # Initialize network if not in context
        if not hasattr(context, 'model'):
            context.model = Net().to(DEVICE)
        
        trainloader, valloader, _ = dataset_loader(partition_id, num_partitions)
        
        return FedPartAvgClient(
            partition_id=partition_id,
            model=context.model,
            train_loader=trainloader,
            val_loader=valloader,
            num_epochs=EPOCHS,
            context=context
        ).to_client()
    return fedpart_avg_client_fn
