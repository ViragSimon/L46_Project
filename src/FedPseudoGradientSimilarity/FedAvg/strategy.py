from ...model import *
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, ndarrays_to_parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays, FitIns, FitRes, EvaluateIns, EvaluateRes, NDArrays
from typing import Optional, List, Tuple, Union
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.client_proxy import ClientProxy
from ...helper import get_parameters_size
from ...FedPart.FedAvg.strategy import FedPartAvg 
from functools import partial, reduce
import numpy as np
import torch


class FedPseudoGradientSimilarityPartAvg(FedPartAvg):
    def __init__(self, aggregate_mode: str = "original", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.earlier_parameters = None
        self.aggregate_mode = aggregate_mode

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        params = super().initialize_parameters(client_manager)
        self.earlier_parameters = self.latest_parameters
        return params

    def pseudo_gradients(self, current_ndarrays: NDArrays, previous_ndarrays: Parameters) -> List[NDArrays]:
        """Calculate pseudo gradients between current and previous parameters"""
        prev_ndarrays = parameters_to_ndarrays(previous_ndarrays)
        pseudo_gradient = [
            current_layer - prev_layer
            for current_layer, prev_layer in zip(current_ndarrays, prev_ndarrays)
        ]
        return pseudo_gradient

    def client_pseudo_gradients(self, global_ndarrays: NDArrays, results: List[Tuple[ClientProxy, FitRes]]) -> List[List[NDArrays]]:
        """Calculate pseudo gradients for each client's update"""
        client_pseudo_gradients = []
        for _, fit_res in results:
            client_ndarrays = parameters_to_ndarrays(fit_res.parameters)
            pseudo_gradient = [
                client_layer - global_layer
                for client_layer, global_layer in zip(client_ndarrays, global_ndarrays)
            ]
            client_pseudo_gradients.append(pseudo_gradient)
        return client_pseudo_gradients

    def _cosine_similarity_of_pseudo_gradients(self, client_pseudo_gradients: List[List[NDArrays]], global_pseudo_gradient: List[NDArrays]) -> List[float]:
        """Calculate cosine similarity between client pseudo gradients and global pseudo gradient"""
        cos = torch.nn.CosineSimilarity(dim=0)
        similarities = []
        
        # Convert global pseudo gradient to flattened tensor
        global_tensor = torch.cat([torch.from_numpy(layer).flatten() for layer in global_pseudo_gradient])
                
        for i, client_gradient in enumerate(client_pseudo_gradients):
            client_tensor = torch.cat([torch.from_numpy(layer).flatten() for layer in client_gradient])
            if torch.norm(global_tensor) < 1e-10 or torch.norm(client_tensor) < 1e-10:
                similarity = 0.0
            else:
                similarity = cos(client_tensor.float(), global_tensor.float()).item()
            
            similarities.append(similarity)
            
        return similarities

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

        # Calculate global pseudo gradient between current and previous global state
        global_pseudo_gradient = self.pseudo_gradients(
            parameters_to_ndarrays(self.latest_parameters),
            self.earlier_parameters
        )

        # Calculate client pseudo gradients between client updates and current global state  
        trained_layer = results[0][1].metrics["trained_layer"]

        if server_round == 1:
            update_direction = self.pseudo_gradient_direction_update(parameters_to_ndarrays(self.latest_parameters),results)
            current_model = parameters_to_ndarrays(self.latest_parameters)
            updated_model = [current_layer + update_layer for current_layer, update_layer in zip(current_model, update_direction)]
            self.earlier_parameters = self.latest_parameters
            self.latest_parameters = ndarrays_to_parameters(updated_model)
            metrics_aggregated = {}

        else:
            
            if trained_layer == -1:

                client_pseudo_gradients = self.client_pseudo_gradients(
                    parameters_to_ndarrays(self.latest_parameters),
                    results
                )

                # Calculate cosine similarities between client and global pseudo gradients
                cos_similarities = self._cosine_similarity_of_pseudo_gradients(
                    client_pseudo_gradients,
                    global_pseudo_gradient
                )
                client_params = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
                aggregated_weights = self._aggregate_params_weighted(client_params, cos_similarities)
                self.earlier_parameters = self.latest_parameters
                self.latest_parameters = ndarrays_to_parameters(aggregated_weights)
            else:
                current_model = parameters_to_ndarrays(self.latest_parameters)
                # Only calculate pseudo gradients for trainable layers
                client_pseudo_gradients = self.client_pseudo_gradients(
                    [current_model[trained_layer*2], current_model[trained_layer*2 + 1]],
                    results
                )

                # Calculate cosine similarities between client and global pseudo gradients for trainable layers
                cos_similarities = self._cosine_similarity_of_pseudo_gradients(
                    client_pseudo_gradients,
                    [global_pseudo_gradient[trained_layer*2], global_pseudo_gradient[trained_layer*2 + 1]]
                )
                client_params = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
                aggregated_weights = self._aggregate_params_weighted(client_params, cos_similarities)
                current_model[trained_layer*2] = aggregated_weights[0]
                current_model[trained_layer*2 + 1] = aggregated_weights[1]

        
                self.earlier_parameters = self.latest_parameters
                self.latest_parameters = ndarrays_to_parameters(current_model)

        self.update_full_client_models(server_round, results, self.previous_parameters)

        self.update_metrics(server_round, self.latest_parameters, results)

        metrics_aggregated = {}
        return self.latest_parameters, metrics_aggregated


    def _rescaled_sigmoid(self, sim: float) -> float:

        sigmoid_at_minus_1 = 1 / (1 + np.exp(1))
        sigmoid_at_1 = 1 / (1 + np.exp(-1))
        standard_sigmoid_val = 1 / (1 + np.exp(-sim))
        return (standard_sigmoid_val - sigmoid_at_minus_1) / (sigmoid_at_1 - sigmoid_at_minus_1)
    
    def _aggregate_params_weighted(self, results: list[NDArrays], cos_similarities: list[float]) -> NDArrays:
        assert len(results) == len(cos_similarities)
        
        if len(cos_similarities) == 0:
            return results[0] if len(results) > 0 else []

        if self.aggregate_mode == "original":
            sum_sim = sum(cos_similarities)
            
            if abs(sum_sim) < 1e-10:
                uniform_weight = 1.0 / len(cos_similarities)
                weighted_weights = [[layer * uniform_weight for layer in results[i]] for i in range(len(results))]
                weights_prime: NDArrays = [reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)]
            else:
                weighted_weights = [[layer * cos_similarities[i] for layer in results[i]] for i in range(len(results))]
                weights_prime: NDArrays = [reduce(np.add, layer_updates)/sum_sim for layer_updates in zip(*weighted_weights)]
        elif self.aggregate_mode == "softmax":
            # Apply softmax to cosine similarities
            exp_similarities = [np.exp(sim) for sim in cos_similarities]
            sum_exp = sum(exp_similarities)
            
            if abs(sum_exp) < 1e-10:
                uniform_weight = 1.0 / len(cos_similarities)
                weighted_weights = [[layer * uniform_weight for layer in results[i]] for i in range(len(results))]
                weights_prime: NDArrays = [reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)]
            else:
                softmax_weights = [exp_sim/sum_exp for exp_sim in exp_similarities]
                weighted_weights = [[layer * softmax_weights[i] for layer in results[i]] for i in range(len(results))]
                weights_prime: NDArrays = [reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)]
        elif self.aggregate_mode == "sigmoid":
            sigmoid_values = [1 / (1 + np.exp(-sim)) for sim in cos_similarities]
            sum_sigmoid = sum(sigmoid_values)
            
            if abs(sum_sigmoid) < 1e-10:
                uniform_weight = 1.0 / len(cos_similarities)
                weighted_weights = [[layer * uniform_weight for layer in results[i]] for i in range(len(results))]
                weights_prime: NDArrays = [reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)]
            else:
                sigmoid_weights = [sig_val / sum_sigmoid for sig_val in sigmoid_values]
                weighted_weights = [[layer * sigmoid_weights[i] for layer in results[i]] for i in range(len(results))]
                weights_prime: NDArrays = [reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)]
        elif self.aggregate_mode == "normalized_sigmoid":
            sigmoid_values = [self._rescaled_sigmoid(sim) for sim in cos_similarities]
            sum_sigmoid = sum(sigmoid_values)
            
            if abs(sum_sigmoid) < 1e-10:
                uniform_weight = 1.0 / len(cos_similarities)
                weighted_weights = [[layer * uniform_weight for layer in results[i]] for i in range(len(results))]
                weights_prime: NDArrays = [reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)]
            else:
                sigmoid_weights = [sig_val / sum_sigmoid for sig_val in sigmoid_values]
                weighted_weights = [[layer * sigmoid_weights[i] for layer in results[i]] for i in range(len(results))]
                weights_prime: NDArrays = [reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)]
        elif self.aggregate_mode == "linear":
            linear_values = [(sim + 1) / 2 for sim in cos_similarities]
            sum_linear = sum(linear_values)
            
            if abs(sum_linear) < 1e-10:
                uniform_weight = 1.0 / len(cos_similarities)
                weighted_weights = [[layer * uniform_weight for layer in results[i]] for i in range(len(results))]
                weights_prime: NDArrays = [reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)]
            else:
                linear_weights = [linear_val / sum_linear for linear_val in linear_values]
                weighted_weights = [[layer * linear_weights[i] for layer in results[i]] for i in range(len(results))]
                weights_prime: NDArrays = [reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)]
        elif self.aggregate_mode == "absolute":
            absolute_values = [abs(sim) for sim in cos_similarities]
            sum_absolute = sum(absolute_values)
            if abs(sum_absolute) < 1e-10:
                uniform_weight = 1.0 / len(cos_similarities)
                weighted_weights = [[layer * uniform_weight for layer in results[i]] for i in range(len(results))]
                weights_prime: NDArrays = [reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)]
            else:
                absolute_weights = [abs_val / sum_absolute for abs_val in absolute_values]
                weighted_weights = [[layer * absolute_weights[i] for layer in results[i]] for i in range(len(results))]
                weights_prime: NDArrays = [reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)]
        else:
            raise ValueError(f"Invalid mode: {self.aggregate_mode}")
        
        return weights_prime
    


    def pseudo_gradient_direction_update(self, global_ndarrays,results):
        num_clients = len(results)
        client_pseudo_gradients = []
        for _, fit_res in results:
            client_ndarrays = parameters_to_ndarrays(fit_res.parameters)

            pseudo_gradient = [
                client_layer - global_layer
                for client_layer, global_layer in zip(client_ndarrays, global_ndarrays)
            ]
            client_pseudo_gradients.append(pseudo_gradient)

        summed_gradients = list(client_pseudo_gradients[0])
        

        for pseudo_gradient in client_pseudo_gradients[1:]:
            for i in range(len(summed_gradients)):
                summed_gradients[i] += pseudo_gradient[i]
                
        update_direction = [grad / num_clients for grad in summed_gradients]
        return update_direction