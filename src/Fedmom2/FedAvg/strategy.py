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


"""
Implementation of this file based on the previous project implementation but has been significantly modified and extended
"""

class FedAvgMom2(FedPartAvg):
    def __init__(self, aggregate_mode: str = "original", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_optim_state = None
        self.fed_mom2_result = {}
        self.fed_mom2_model_results = {}
        self.aggregate_mode = aggregate_mode

    def get_results(self):
        return self.fed_mom2_result, self.fed_mom2_model_results, self.metrics_history


    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        model = Net()
        ndarrays = get_parameters(model)
        self.layer_training_sequence = self.generate_layer_training_sequence()
        self.number_of_layers = len(ndarrays)
        self.latest_parameters = ndarrays_to_parameters(ndarrays)
        self.previous_parameters = self.latest_parameters

        return ndarrays_to_parameters(ndarrays)
    
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:

        if self.evaluate_fn is None:
            return None
        
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        
        if server_round in self.fed_mom2_model_results:  
            expand_fed_part_avg_model_results= {**self.fed_mom2_model_results[server_round], "global_loss": eval_res[0], "global_metrics": eval_res[1]}
        else:
            expand_fed_part_avg_model_results= {"global_loss": eval_res[0], "global_metrics": eval_res[1]}
        
        self.fed_mom2_model_results[server_round] = expand_fed_part_avg_model_results
        
        loss, metrics = eval_res
        return loss, metrics
    


    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        
        config = {"trainable_layers": self.layer_training_sequence[self.training_sequence_index], "updated_layers": self.updated_layers}

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
    


    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        if self.fraction_evaluate == 0.0:
            return []
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]
    


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
        
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        metrics = {}
        optimizer_states_serialized = [res.metrics.get("optimizer_state", None) for _, res in results]
        optimizer_states = [deserialize_optimizer_state(state) for state in optimizer_states_serialized if state is not None]

        if server_round == 1:
            equal_weights = [1] * len(optimizer_states)
            prev_global_optimizer_state = self._aggregate_optimizer_states_weighted(optimizer_states, equal_weights)
        else:
            prev_global_optimizer_state = deserialize_optimizer_state(self.global_optim_state)
        cos_similarities = self._cos_similarity_from_optimizer_states(optimizer_states, prev_global_optimizer_state)

        aggregated_weights = self._aggregate_params_weighted(weights_results, cos_similarities)
        trained_layer = results[0][1].metrics.get("trained_layer", -1)

        if trained_layer == -1:
            self.latest_parameters = ndarrays_to_parameters(aggregated_weights)
        else:
            current_model = parameters_to_ndarrays(self.latest_parameters)
            current_model[trained_layer* 2] = aggregated_weights[0]
            current_model[trained_layer* 2 +1] = aggregated_weights[1]
            self.latest_parameters = ndarrays_to_parameters(current_model)

        aggregated_optimizer_state = self._aggregate_optimizer_states_weighted(optimizer_states, cos_similarities)
        aggregated_optimizer_state_serialized = serialize_optimizer_state(aggregated_optimizer_state)
        metrics["optimizer_state"] = aggregated_optimizer_state_serialized
        self.global_optim_state = aggregated_optimizer_state_serialized

        sizes = sum([sys.getsizeof(data) for data in optimizer_states_serialized if data is not None])
        total_size += sizes

        if self.fed_mom2_result.get(server_round):
            self.fed_mom2_result[server_round]["total_size"] = total_size
        else:
            self.fed_mom2_result[server_round] = {"total_size": total_size}


        self.update_full_client_models(server_round, results, self.previous_parameters)

        self.update_metrics(server_round, self.latest_parameters, results)

        return self.latest_parameters, metrics


    def _aggregate_optimizer_states_weighted(self, optimizer_states, similarities):
        aggregated_state_dict = copy.deepcopy(optimizer_states[0])
        weights = self.get_similarity_weights(similarities)
        sum_weights = sum(weights)
        
        for layer in range(len(optimizer_states[0]['state'])):
            weighted_fst_momentum = torch.zeros_like(optimizer_states[0]['state'][layer]['exp_avg'])
            weighted_snd_momentum = torch.zeros_like(optimizer_states[0]['state'][layer]['exp_avg_sq'])
            
            # Accumulate weighted momentums from each client
            for i in range(len(optimizer_states)):
                weight = weights[i] / sum_weights
                weighted_fst_momentum += optimizer_states[i]['state'][layer]['exp_avg'] * weight
                weighted_snd_momentum += optimizer_states[i]['state'][layer]['exp_avg_sq'] * weight
            weighted_fst_momentum = weighted_fst_momentum 
            weighted_snd_momentum = weighted_snd_momentum 
            aggregated_state_dict['state'][layer]['exp_avg'] = weighted_fst_momentum
            aggregated_state_dict['state'][layer]['exp_avg_sq'] = weighted_snd_momentum
        max_steps = max([optimizer_states[i]['state'][layer]['step'] for i in range(len(optimizer_states))])
        aggregated_state_dict['state'][layer]['step'] = max_steps
        
        return aggregated_state_dict
    
    def _rescaled_sigmoid(self, sim: float) -> float:

        sigmoid_at_minus_1 = 1 / (1 + np.exp(1))
        sigmoid_at_1 = 1 / (1 + np.exp(-1))
        standard_sigmoid_val = 1 / (1 + np.exp(-sim))
        return (standard_sigmoid_val - sigmoid_at_minus_1) / (sigmoid_at_1 - sigmoid_at_minus_1)

    def get_similarity_weights(self, similarities):
        if self.aggregate_mode == "original":
            return similarities
        elif self.aggregate_mode == "softmax":
            return [np.exp(sim) / np.sum(np.exp(similarities)) for sim in similarities]
        elif self.aggregate_mode == "sigmoid":
            return [1 / (1 + np.exp(-sim)) for sim in similarities]
        elif self.aggregate_mode == "normalized_sigmoid":
            return [self._rescaled_sigmoid(sim) for sim in similarities]
        elif self.aggregate_mode == "linear":
            return [(sim + 1) / 2 for sim in similarities]
        elif self.aggregate_mode == "absolute":
            return [abs(sim) for sim in similarities]
        else:
            raise ValueError(f"Invalid mode: {self.aggregate_mode}")

    def _aggregate_params_weighted(self, results: list[tuple[NDArrays, int]], cos_similarities: list[float]) -> NDArrays:
        assert len(results) == len(cos_similarities)

        if self.aggregate_mode == "original":
            sum_sim = sum(cos_similarities)
            weighted_weights = [[layer * cos_similarities[i] for layer in results[i][0]] for i in range(len(results))]
            weights_prime: NDArrays = [reduce(np.add, layer_updates)/sum_sim for layer_updates in zip(*weighted_weights)]
        elif self.aggregate_mode == "softmax":
            # Apply softmax to cosine similarities
            exp_similarities = [np.exp(sim) for sim in cos_similarities]
            sum_exp = sum(exp_similarities)
            softmax_weights = [exp_sim/sum_exp for exp_sim in exp_similarities]
            weighted_weights = [[layer * softmax_weights[i] for layer in results[i][0]] for i in range(len(results))]
            weights_prime: NDArrays = [reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)]
        elif self.aggregate_mode == "sigmoid":
            sigmoid_values = [1 / (1 + np.exp(-sim)) for sim in cos_similarities]
            sum_sigmoid = sum(sigmoid_values)
            sigmoid_weights = [sig_val / sum_sigmoid for sig_val in sigmoid_values]
            weighted_weights = [[layer * sigmoid_weights[i] for layer in results[i][0]] for i in range(len(results))]
            weights_prime: NDArrays = [reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)]
        elif self.aggregate_mode == "normalized_sigmoid":
            sigmoid_values = [self._rescaled_sigmoid(sim) for sim in cos_similarities]
            sum_sigmoid = sum(sigmoid_values)
            sigmoid_weights = [sig_val / sum_sigmoid for sig_val in sigmoid_values]
            weighted_weights = [[layer * sigmoid_weights[i] for layer in results[i][0]] for i in range(len(results))]
            weights_prime: NDArrays = [reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)]
        elif self.aggregate_mode == "linear":
            linear_values = [(sim + 1) / 2 for sim in cos_similarities]
            sum_linear = sum(linear_values)
            linear_weights = [linear_val / sum_linear for linear_val in linear_values]
            weighted_weights = [[layer * linear_weights[i] for layer in results[i][0]] for i in range(len(results))]
            weights_prime: NDArrays = [reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)]
        elif self.aggregate_mode == "absolute":
            absolute_values = [abs(sim) for sim in cos_similarities]
            sum_absolute = sum(absolute_values)
            absolute_weights = [abs_val / sum_absolute for abs_val in absolute_values]
            weighted_weights = [[layer * absolute_weights[i] for layer in results[i][0]] for i in range(len(results))]
            weights_prime: NDArrays = [reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)]
        else:
            raise ValueError(f"Invalid mode: {self.aggregate_mode}")


        return weights_prime

    def _cos_similarity_from_optimizer_states(self, optimizer_states, prev_global_optimizer_state):

        
        cos = torch.nn.CosineSimilarity(dim=-1)
        similarities = []
        layers = range(len(optimizer_states[0]['state']))

     
        fst_momentum_glob = [prev_global_optimizer_state['state'][l]['exp_avg'].view(-1) for l in layers]
        snd_momentum_glob = [prev_global_optimizer_state['state'][l]['exp_avg_sq'].view(-1) for l in layers]
        glob_state_vec = torch.cat(fst_momentum_glob + snd_momentum_glob)

        for client in range(len(optimizer_states)):
            fst_momentum_client = [optimizer_states[client]['state'][l]['exp_avg'].view(-1) for l in layers]
            snd_momentum_client = [optimizer_states[client]['state'][l]['exp_avg_sq'].view(-1) for l in layers]
            client_state_vec = torch.cat(fst_momentum_client + snd_momentum_client)
            sim = cos(client_state_vec, glob_state_vec)
            similarities.append(sim.item())
      
        
        return similarities

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

        if self.fed_mom2_result.get(server_round):
            self.fed_mom2_result[server_round]["total_loss"] = total_loss
        else:
            self.fed_mom2_result[server_round] = {"total_loss": total_loss}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated