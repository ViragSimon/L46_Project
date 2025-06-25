# local_adam_strategy.py

from src.FedPart.FedAvg.strategy import FedPartAvg
from flwr.common import Parameters, ndarrays_to_parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays, FitIns, FitRes, EvaluateIns, EvaluateRes
from typing import Optional, List, Tuple, Union, Dict
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.client_proxy import ClientProxy
from ...helper import get_parameters_size
from ...Original.FedAvg.strategy import CustomFedAvg
from ...model import *
import numpy as np


class FedPartAdam(FedPartAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.u_vectors = []
        self.v_vectors = []

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize parameters for LocalAdam with model parameters plus optimizer state."""
        net = Net()
        ndarrays = get_parameters(net)
        self.layer_training_sequence = self.generate_layer_training_sequence()
        self.number_of_layers = len(ndarrays)
        
        
        # For LocalAdam, we need model parameters plus zero-initialized u and v vectors
        self.u_vectors = [np.zeros_like(p) for p in ndarrays]
        self.v_vectors = [np.zeros_like(p) for p in ndarrays]
        full_params = ndarrays + self.u_vectors + self.v_vectors
        
        # Store only model params for evaluation (not optimizer state)
        self.latest_parameters = ndarrays_to_parameters(ndarrays)
        self.previous_parameters = self.latest_parameters
        
        print(f"[DEBUG] Initialized parameters with {len(ndarrays)} model parameters")
        print(f"[DEBUG] Model architecture: {[p.shape for p in ndarrays]}")
        
        return ndarrays_to_parameters(full_params)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        config = {
            "trainable_layers": self.layer_training_sequence[self.training_sequence_index],
            "updated_layers": self.updated_layers,
        }
        
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        
        # For LocalAdam, we need to ensure parameters contain [model_params, u_vectors, v_vectors]
        # If this is the first round or parameters don't have the correct format, initialize them
        fit_configurations = []
        params_array = parameters_to_ndarrays(parameters)

        if self.layer_training_sequence[self.training_sequence_index] == -1 or self.updated_layers == -1:
            selected_params = ndarrays_to_parameters(params_array)
            u_vectors = self.u_vectors
            v_vectors = self.v_vectors
            final_params = ndarrays_to_parameters(params_array + u_vectors + v_vectors)
        else:
            layer_idx = self.updated_layers
            selected_params = [
                    params_array[layer_idx * 2],     # Weight
                    params_array[layer_idx * 2 + 1]  # Bias
                ]
            u_vectors = [self.u_vectors[layer_idx * 2], self.u_vectors[layer_idx * 2 + 1]]
            v_vectors = [self.v_vectors[layer_idx * 2], self.v_vectors[layer_idx * 2 + 1]]
            final_params = ndarrays_to_parameters(selected_params + u_vectors + v_vectors)


        for idx, client in enumerate(clients):
            fit_configurations.append((client, FitIns(final_params, config)))

        self.updated_layers = self.layer_training_sequence[self.training_sequence_index]
        self.training_sequence_index = self.training_sequence_index + 1
        
        return fit_configurations
    
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
        
        if self.fed_part_avg_result.get(server_round):
            self.fed_part_avg_result[server_round].update({"total_size": total_size})
        else:
            self.fed_part_avg_result[server_round] = {"total_size": total_size}

        weights_results, u_results, v_results = [], [], []
        cleaned_results = []
        for client, fit_res in results:
            all_ndarrays = parameters_to_ndarrays(fit_res.parameters)
            num_examples = fit_res.num_examples
            num_weight_params = int(fit_res.metrics["num_weight_params"])

            # For LocalAdam, clients return full model parameters plus optimizer state
            # Split the returned parameters into weights, u, and v
            weights = all_ndarrays[0:num_weight_params]
            u = all_ndarrays[num_weight_params : 2 * num_weight_params]
            v = all_ndarrays[2 * num_weight_params : 3 * num_weight_params]
            cleaned_results.append((client, FitRes(parameters=ndarrays_to_parameters(weights), num_examples=fit_res.num_examples, metrics=fit_res.metrics, status=fit_res.status)))

            weights_results.append((weights, num_examples))
            u_results.append((u, num_examples))
            v_results.append((v, num_examples))
        
        # Aggregate each component
        aggregated_weights = aggregate(weights_results)
        aggregated_u = aggregate(u_results)
        aggregated_v = aggregate(v_results)
        
        # Update the server's copy of the full model
        trained_layer = int(results[0][1].metrics["trained_layer"])
        if trained_layer == -1:
            # Full model update
            self.latest_parameters = ndarrays_to_parameters(aggregated_weights)
            self.u_vectors = aggregated_u
            self.v_vectors = aggregated_v
        else:
            # Partial model update - only update the specific layer
            current_model = parameters_to_ndarrays(self.latest_parameters)
            current_model[trained_layer*2] = aggregated_weights[trained_layer*2]
            current_model[trained_layer*2 + 1] = aggregated_weights[trained_layer*2 + 1]
            self.latest_parameters = ndarrays_to_parameters(current_model)
            self.u_vectors[trained_layer*2] = aggregated_u[0]
            self.u_vectors[trained_layer*2 + 1] = aggregated_u[1]
            self.v_vectors[trained_layer*2] = aggregated_v[0]
            self.v_vectors[trained_layer*2 + 1] = aggregated_v[1]

        cleaned_results =[]


        self.update_full_client_models(server_round, cleaned_results, self.previous_parameters)

        self.update_metrics(server_round, self.latest_parameters, cleaned_results)

        
        return self.latest_parameters, {}

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        
        # For LocalAdam, parameters contain [model_params, u_vectors, v_vectors]
        # We need to extract only the model parameters for evaluation
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        
        # Get the number of model parameters (excluding optimizer state)
        net = Net()
        num_model_params = len(get_parameters(net))
        
        # Extract only the model parameters
        model_parameters = parameters_ndarrays[:num_model_params]
        
        eval_res = self.evaluate_fn(server_round, model_parameters, {})
        if eval_res is None:
            return None
        
        if server_round in self.fed_part_avg_model_results:  
            expand_fed_part_avg_model_results= {**self.fed_part_avg_model_results[server_round], "global_loss": eval_res[0], "global_metrics": eval_res[1]}
        else:
            expand_fed_part_avg_model_results= {"global_loss": eval_res[0], "global_metrics": eval_res[1]}
        
        self.fed_part_avg_model_results[server_round] = expand_fed_part_avg_model_results
    
        loss, metrics = eval_res
        return loss, metrics