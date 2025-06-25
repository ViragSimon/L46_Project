from ...model import *
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, ndarrays_to_parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays, FitIns, FitRes, EvaluateIns, EvaluateRes
from typing import Optional, List, Tuple, Union
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.client_proxy import ClientProxy
from ...helper import get_parameters_size
from collections import defaultdict



"""
Implementation of this file based on the previous project implementation but has been significantly modified and extended
"""

class CustomFedAvg(FedAvg):

    def __init__(self, initial_parameters: Optional[Parameters] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fed_avg_result ={}
        self.fed_avg_model_results = {}
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

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        net = Net().to(DEVICE)
        ndarrays = get_parameters(net)
        self.previous_parameters = ndarrays_to_parameters(ndarrays)

        return ndarrays_to_parameters(ndarrays)
    
    def get_results(self):
        return self.fed_avg_result, self.fed_avg_model_results, self.metrics_history
    


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


    def update_full_client_models(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]]):
        """Update the stored full client models by reconstructing them from partial updates."""
        for client, fit_res in results:
            client_id = client.cid
            self.full_client_models[client_id] = fit_res.parameters
            
    
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

        if server_round in self.fed_avg_model_results:
            eval_results = {**self.fed_avg_model_results[server_round], **eval_results}

        self.fed_avg_model_results[server_round] = eval_results

        return loss, metrics
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        config = {}
        
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
    
        fit_ins = FitIns(parameters, config)
        fit_configurations = [(client, fit_ins) for client in clients]
        
        return fit_configurations

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []

        # Create evaluation configuration and instructions
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients for evaluation
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
        """Aggregate fit results using weighted average."""
        # Calculate total parameter size
        total_size = 0
        for _, fit_res in results:
            total_size += get_parameters_size(fit_res.parameters)
            total_size += fit_res.metrics["recieved_parameter_size"]

        # Update round results
        expand_fed_avg_result = {"total_size": total_size}
        if server_round in self.fed_avg_result:
            expand_fed_avg_result = {**self.fed_avg_result[server_round], **expand_fed_avg_result}
        self.fed_avg_result[server_round] = expand_fed_avg_result

        # Aggregate parameters
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        self.update_full_client_models(server_round, results)

        self.update_metrics(server_round, parameters_aggregated, results)
        
        return parameters_aggregated, {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # Calculate total loss
        total_loss = sum(evaluate_res.loss for _, evaluate_res in results)

        # Update round results
        expand_fed_avg_result = {"total_loss": total_loss}
        if server_round in self.fed_avg_result:
            expand_fed_avg_result = {**self.fed_avg_result[server_round], **expand_fed_avg_result}
        self.fed_avg_result[server_round] = expand_fed_avg_result

        # Aggregate losses
        loss_aggregated = weighted_loss_avg([
            (evaluate_res.num_examples, evaluate_res.loss)
            for _, evaluate_res in results
        ])

        return loss_aggregated, {}
