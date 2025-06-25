from flwr.common import NDArrays, Scalar
import sys
from ...model import *
from flwr.client import Client, ClientApp, NumPyClient
import copy
import numpy as np
from typing import Callable, Dict, Optional, Tuple
from collections import OrderedDict
from ...dataset import load_datasets
from flwr.common import Context
from ...model import set_parameters, get_parameters, fedAvg_train, test, DEVICE, EPOCHS

class FedProxClient(NumPyClient):
    def __init__(self, partition_id, model, train_loader, val_loader, epochs: int):
        self.partition_id = partition_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs

    def get_parameters(self, config):
        print(f"[Client {self.partition_id}] get_parameters")
        return get_parameters(self.model)

    def fit(self, parameters, config):
        print(f"[Client {self.partition_id}] fit, config: {config}")
        set_parameters(self.model, parameters)
        global_params = copy.deepcopy(self.model).parameters()
        fedProx_train(self.model, self.train_loader, self.epochs, config["proximal_mu"], global_params)
        return get_parameters(self.model), len(self.train_loader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.partition_id}] evaluate, config: {config}")
        set_parameters(self.model, parameters)
        loss, accuracy = test(self.model, self.val_loader)
        return float(loss), len(self.val_loader), {"accuracy": float(accuracy)}


def get_fedprox_client_fn(dataset_loader: Callable) -> Callable:
    def fedprox_client_fn(context: Context) -> Client:
        model = Net().to(DEVICE)
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        train_loader, val_loader, _ = dataset_loader(partition_id, num_partitions)
        return FedProxClient(partition_id, model, train_loader, val_loader, EPOCHS).to_client()
    return fedprox_client_fn