from ...model import *
from flwr.common import Parameters, ndarrays_to_parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays, FitIns, FitRes, EvaluateIns, EvaluateRes
from typing import Optional, List, Tuple, Union
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.client_proxy import ClientProxy
from ...helper import get_parameters_size
from ...FedPart.FedAvg.strategy import FedPartAvg


"""
Implementation of this file based on the previous project implementation but has been greatly modified and extneded
"""

class FedPartProx(FedPartAvg):
    def __init__(self, proximal_mu: float, initial_parameters: Optional[Parameters] = None, *args, **kwargs):
        super().__init__(initial_parameters, *args, **kwargs)
        self.proximal_mu = proximal_mu
        self.fed_part_prox_result = {}
        self.fed_part_prox_model_results = {}

    def get_results(self):
        return self.fed_part_prox_result, self.fed_part_prox_model_results, self.metrics_history


    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:

        client_config_pairs = super().configure_fit(server_round, parameters, client_manager)

        return [
            (client, FitIns(fit_ins.parameters, {**fit_ins.config, "proximal_mu": self.proximal_mu})) for client, fit_ins in client_config_pairs
        ]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        
        total_size = 0
        for client, fit_res in results:
            total_size += get_parameters_size(fit_res.parameters)
            total_size += fit_res.metrics["recieved_parameter_size"]

        print(f"total size: {total_size}")
        
        if self.fed_part_prox_result.get(server_round):
            self.fed_part_prox_result[server_round]["total_size"] = total_size
        else:
            self.fed_part_prox_result[server_round] = {"total_size": total_size}
        

        weights_results = [(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results]

        aggregated_weights = aggregate(weights_results)
        trained_layer = results[0][1].metrics["trained_layer"]


        if self.layer_training_sequence[self.training_sequence_index -1] == -1:
            self.latest_parameters = ndarrays_to_parameters(aggregated_weights)
        else:
            current_model = parameters_to_ndarrays(self.latest_parameters)
            print(f"updateing layers {self.layer_training_sequence[self.training_sequence_index -1]* 2}  and {self.layer_training_sequence[self.training_sequence_index -1]* 2 + 1} ")
            current_model[self.layer_training_sequence[self.training_sequence_index -1]* 2] = aggregated_weights[0]
            current_model[self.layer_training_sequence[self.training_sequence_index -1]* 2 +1] = aggregated_weights[1]
            self.latest_parameters = ndarrays_to_parameters(current_model)

        self.update_full_client_models(server_round, results, self.previous_parameters)

        self.update_metrics(server_round, self.latest_parameters, results)


        metrics_aggregated = {}
        return self.latest_parameters, metrics_aggregated
    

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

        if self.fed_part_prox_result.get(server_round):
            self.fed_part_prox_result[server_round]["total_loss"] = total_loss
        else:
            self.fed_part_prox_result[server_round] = {"total_loss": total_loss}

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
        
        if server_round in self.fed_part_prox_model_results:  
            expand_fed_part_prox_model_results= {**self.fed_part_prox_model_results[server_round], "global_loss": eval_res[0], "global_metrics": eval_res[1]}
        else:
            expand_fed_part_prox_model_results= {"global_loss": eval_res[0], "global_metrics": eval_res[1]}
        
        self.fed_part_prox_model_results[server_round] = expand_fed_part_prox_model_results
        
        loss, metrics = eval_res
        return loss, metrics

        


        
