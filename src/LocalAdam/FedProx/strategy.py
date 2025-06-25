from ...model import *
from flwr.common import Parameters, ndarrays_to_parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays, FitIns, FitRes, EvaluateIns, EvaluateRes
from typing import Optional, List, Tuple, Union
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.client_proxy import ClientProxy
from ...helper import get_parameters_size
from ...LocalAdam.FedAvg.strategy import FedPartAdam


class FedPartProxAdam(FedPartAdam):
    def __init__(self, proximal_mu: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proximal_mu = proximal_mu
        self.global_params = None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        client_config_pairs = super().configure_fit(server_round, parameters, client_manager)

        return [
            (client, FitIns(fit_ins.parameters, {**fit_ins.config, "proximal_mu": self.proximal_mu})) for client, fit_ins in client_config_pairs
        ]
        


        
