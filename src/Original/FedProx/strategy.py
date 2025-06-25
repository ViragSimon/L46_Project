from ...model import *
from ...Original.FedAvg.strategy import CustomFedAvg
from flwr.common import Parameters, ndarrays_to_parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays, FitIns, FitRes, EvaluateIns, EvaluateRes
from typing import Optional, List, Tuple, Union
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.client_proxy import ClientProxy
from ...helper import get_parameters_size


class CustomFedProx(CustomFedAvg):
    def __init__(self, proximal_mu: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proximal_mu = proximal_mu
        self.fed_prox_result = {}
        self.fed_prox_model_results = {}

    def get_results(self):
        return self.fed_prox_result, self.fed_prox_model_results, self.metrics_history

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        client_config_pairs = super().configure_fit(
            server_round, parameters, client_manager
        )

        return [
            (
                client,
                FitIns(
                    fit_ins.parameters,
                    {**fit_ins.config, "proximal_mu": self.proximal_mu},
                ),
            )
            for client, fit_ins in client_config_pairs
        ]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        total_size = 0
        for client, fit_res in results:
            total_size += get_parameters_size(fit_res.parameters) *2
        print(f"total size: {total_size}")
        
        if self.fed_prox_result.get(server_round):
            self.fed_prox_result[server_round]["total_size"] = total_size
        else:
            self.fed_prox_result[server_round] = {"total_size": total_size}
        

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        self.update_full_client_models(server_round, results)

        self.update_metrics(server_round, parameters_aggregated, results)

        metrics_aggregated = {}
        return parameters_aggregated, metrics_aggregated


    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        
        if not results:
            return None, {}
        
        total_loss = 0
        for _, evaluate_res in results:
            total_loss += evaluate_res.loss

        if self.fed_prox_result.get(server_round):
            self.fed_prox_result[server_round]["total_loss"] = total_loss
        else:
            self.fed_prox_result[server_round] = {"total_loss": total_loss}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated
    

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        if self.evaluate_fn is None:
            return None

        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        
        loss, metrics = eval_res
        eval_results = {
            "global_loss": loss,
            "global_metrics": metrics
        }

        if server_round in self.fed_prox_model_results:
            eval_results = {**self.fed_prox_model_results[server_round], **eval_results}

        self.fed_prox_model_results[server_round] = eval_results

        return loss, metrics
