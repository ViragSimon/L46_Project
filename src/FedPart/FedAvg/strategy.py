from ...model import *
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, ndarrays_to_parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays, FitIns, FitRes, EvaluateIns, EvaluateRes
from typing import Optional, List, Tuple, Union, Dict
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.client_proxy import ClientProxy
from ...helper import get_parameters_size
from ...Original.FedAvg.strategy import CustomFedAvg
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import os



"""
Implementation of this file based on the previous project implementation but has been significantly modified and extended
"""


class FedPartAvg(CustomFedAvg):
    def __init__(self,initial_parameters: Optional[Parameters] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_parameters = initial_parameters
        self.current_layer = 0 
        self.number_of_layers = None
        self.layer_training_sequence = []
        self.training_sequence_index = 0
        self.latest_parameters = initial_parameters
        self.updated_layers = -1
        self.fed_part_avg_result = {}
        self.fed_part_avg_model_results = {}
        
        self.metrics_history = {
            'l2_norms': [],
            'parameter_differences': [],
            'cosine_similarities': [],
            'client_divergence': [],
            'client_to_central_similarities': [],
            'rounds': []
        }
        self.previous_parameters = initial_parameters
        self.client_parameters_history = defaultdict(list)

        self.full_client_models = {} 

    def get_results(self):
        return self.fed_part_avg_result, self.fed_part_avg_model_results, self.metrics_history

    def calculate_l2_norm(self, parameters: Parameters) -> float:
        """Calculate L2 norm of model parameters."""
        params_array = parameters_to_ndarrays(parameters)
        total_norm = 0.0
        for param in params_array:
            total_norm += np.sum(param ** 2)
        return np.sqrt(total_norm)

    def calculate_parameter_difference(self, current_params: Parameters, previous_params: Parameters) -> float:
        """Calculate parameter difference between rounds."""
        if previous_params is None:
            return 0.0
        
        current_array = parameters_to_ndarrays(current_params)
        previous_array = parameters_to_ndarrays(previous_params)
        
        total_diff = 0.0
        for curr, prev in zip(current_array, previous_array):
            total_diff += np.sum((curr - prev) ** 2)
        return np.sqrt(total_diff)

    def calculate_cosine_similarity(self, current_params: Parameters, previous_params: Parameters) -> float:
        """Calculate cosine similarity between model parameters."""
        if previous_params is None:
            return 1.0
        
        current_array = parameters_to_ndarrays(current_params)
        previous_array = parameters_to_ndarrays(previous_params)
        
        # Flatten all parameters
        current_flat = np.concatenate([param.flatten() for param in current_array])
        previous_flat = np.concatenate([param.flatten() for param in previous_array])
        
        # Calculate cosine similarity
        dot_product = np.dot(current_flat, previous_flat)
        norm_current = np.linalg.norm(current_flat)
        norm_previous = np.linalg.norm(previous_flat)
        
        if norm_current == 0 or norm_previous == 0:
            return 0.0
        
        return dot_product / (norm_current * norm_previous)


    def calculate_client_divergence(self, results: List[Tuple[ClientProxy, FitRes]]) -> float:
        """Calculate model divergence between clients."""
        if len(results) < 2:
            return 0.0
        
        client_params = []
        for _, fit_res in results:
            params_array = parameters_to_ndarrays(fit_res.parameters)
            # Flatten parameters for comparison
            flat_params = np.concatenate([param.flatten() for param in params_array])
            client_params.append(flat_params)
        
        # Calculate pairwise distances
        total_divergence = 0.0
        count = 0
        
        for i in range(len(client_params)):
            for j in range(i + 1, len(client_params)):
                diff = client_params[i] - client_params[j]
                distance = np.linalg.norm(diff)
                total_divergence += distance
                count += 1
        
        return total_divergence / count if count > 0 else 0.0

    def calculate_client_to_central_similarities(self, results: List[Tuple[ClientProxy, FitRes]], central_parameters: Parameters) -> List[float]:
        """Calculate cosine similarity between each client's full model and the central model."""
        if not results:
            return []
        
        # Use stored full client models for comparison
        central_array = parameters_to_ndarrays(central_parameters)
        central_flat = np.concatenate([param.flatten() for param in central_array])
        central_norm = np.linalg.norm(central_flat)
        
        client_similarities = []
        
        for client, fit_res in results:
            client_id = client.cid
            
            # Get the full client model from storage
            if client_id in self.full_client_models:
                client_full_params = self.full_client_models[client_id]
                client_array = parameters_to_ndarrays(client_full_params)
                client_flat = np.concatenate([param.flatten() for param in client_array])
                client_norm = np.linalg.norm(client_flat)
                
                # Calculate cosine similarity
                if central_norm == 0 or client_norm == 0:
                    similarity = 0.0
                else:
                    dot_product = np.dot(central_flat, client_flat)
                    similarity = dot_product / (central_norm * client_norm)
                
                client_similarities.append(similarity)
            else:
                # If we don't have the full model, use partial similarity as fallback
                client_similarities.append(0.0)
        
        return client_similarities

    def update_metrics(self, server_round: int, parameters: Parameters, results: List[Tuple[ClientProxy, FitRes]]):
        """Update all metrics for the current round."""
        l2_norm = self.calculate_l2_norm(parameters)
        
        param_diff = self.calculate_parameter_difference(parameters, self.previous_parameters)
        
        cosine_sim = self.calculate_cosine_similarity(parameters, self.previous_parameters)
        
        client_div = self.calculate_client_divergence(results)
        
        client_to_central_sims = self.calculate_client_to_central_similarities(results, parameters)
        
        self.metrics_history['rounds'].append(server_round)
        self.metrics_history['l2_norms'].append(l2_norm)
        self.metrics_history['parameter_differences'].append(param_diff)
        self.metrics_history['cosine_similarities'].append(cosine_sim)
        self.metrics_history['client_divergence'].append(client_div)
        self.metrics_history['client_to_central_similarities'].append(client_to_central_sims)
        
        for client, fit_res in results:
            self.client_parameters_history[server_round].append(fit_res.parameters)
        
        self.previous_parameters = parameters

    def generate_layer_training_sequence(self) -> List[int]:
        layer_training_sequence = []
        for _ in range(NUM_OF_CYCLES):
            for _ in range(NUM_OF_FULL_UPDATES_BETWEEN_CYCLES):
                    layer_training_sequence.append(-1)
            for layer in range(NETWORK_LEN):
                    layer_training_sequence.append(layer)
                    layer_training_sequence.append(layer)

        return layer_training_sequence
    

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        net = Net()
        ndarrays = get_parameters(net)
        self.layer_training_sequence = self.generate_layer_training_sequence()
        self.number_of_layers = len(ndarrays)
        self.latest_parameters = ndarrays_to_parameters(ndarrays)
        self.previous_parameters = self.latest_parameters
        return ndarrays_to_parameters(ndarrays)
    

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        
        if server_round in self.fed_part_avg_model_results:  
            expand_fed_part_avg_model_results= {**self.fed_part_avg_model_results[server_round], "global_loss": eval_res[0], "global_metrics": eval_res[1]}
        else:
            expand_fed_part_avg_model_results= {"global_loss": eval_res[0], "global_metrics": eval_res[1]}
        
        self.fed_part_avg_model_results[server_round] = expand_fed_part_avg_model_results
    
        loss, metrics = eval_res
        return loss, metrics
    

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        
        config = {"trainable_layers": self.layer_training_sequence[self.training_sequence_index], "updated_layers": self.updated_layers}
        
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        fit_configurations = []

        params_array = parameters_to_ndarrays(parameters)
        

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

        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        
        
        total_size = 0
        for client, fit_res in results:
            total_size += get_parameters_size(fit_res.parameters)
            total_size += fit_res.metrics["recieved_parameter_size"]
            
        print(f"total size: {total_size}")
        
        if self.fed_part_avg_result.get(server_round):
            self.fed_part_avg_result[server_round]["total_size"] = total_size
        else:
            self.fed_part_avg_result[server_round] = {"total_size": total_size}
        


        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        

        aggregated_weights = aggregate(weights_results)
        trained_layer = results[0][1].metrics["trained_layer"]
        print(f"aggregated weight size {len(aggregated_weights)} ")

        if trained_layer == -1:
            self.latest_parameters = ndarrays_to_parameters(aggregated_weights)
        else:
            current_model = parameters_to_ndarrays(self.latest_parameters)
            print(f"updateing layers {trained_layer* 2}  and {trained_layer* 2 + 1} ")
            current_model[trained_layer* 2] = aggregated_weights[0]
            current_model[trained_layer* 2 +1] = aggregated_weights[1]
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
        """Aggregate evaluation losses using weighted average."""

        if not results:
            return None, {}
        
        total_loss = 0
        for _, evaluate_res in results:
            total_loss += evaluate_res.loss

        if self.fed_part_avg_result.get(server_round):
            self.fed_part_avg_result[server_round]["total_loss"] = total_loss
        else:
            self.fed_part_avg_result[server_round] = {"total_loss": total_loss}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated
        
    def update_full_client_models(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], global_parameters: Parameters):
        """Update the stored full client models by reconstructing them from partial updates."""
        
        for client, fit_res in results:
            client_id = client.cid
            trained_layer = fit_res.metrics["trained_layer"]
            
        
            current_full_array = parameters_to_ndarrays(global_parameters)
            

            if trained_layer == -1:

                self.full_client_models[client_id] = fit_res.parameters
            else:
                received_params = parameters_to_ndarrays(fit_res.parameters)
                current_full_array[trained_layer * 2] = received_params[0]  # Weight
                current_full_array[trained_layer * 2 + 1] = received_params[1]  # Bias
                self.full_client_models[client_id] = ndarrays_to_parameters(current_full_array)
        
        