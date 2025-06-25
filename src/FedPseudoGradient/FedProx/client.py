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
from ...FedPart.FedProx.client import FedPartProxClient


class FedPseudoGradientPartProxClient(FedPartProxClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def get_fedpseudo_gradient_part_prox_client_fn(dataset_loader: Callable) -> Callable:
    def fedpseudo_gradient_part_prox_client_fn(context: Context) -> Client:
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        if not hasattr(context, 'model'):
            context.model = Net().to(DEVICE)
        
        train_loader, val_loader, _ = dataset_loader(partition_id, num_partitions)
        
        return FedPseudoGradientPartProxClient(
            partition_id=partition_id,
            model=context.model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=EPOCHS,
            context=context
        ).to_client()
    return fedpseudo_gradient_part_prox_client_fn
        
