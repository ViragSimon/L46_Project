import numpy as np
from ...model import *
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, ndarrays_to_parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays, FitIns, FitRes, EvaluateIns, EvaluateRes
from typing import Optional, List, Tuple, Union
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.client_proxy import ClientProxy
from ...helper import get_parameters_size
from ...FedPart.FedAvg.strategy import FedPartAvg 




class FedPseudoGradientPartAvg(FedPartAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def pseudo_gradient_update(self, global_ndarrays,results):
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
        

        # pseudo_gradient_results = [parameters_to_ndarrays(fit_res.parameters) - parameters_to_ndarrays(self.latest_parameters) for _, fit_res in results]
        # update_direction = sum(pseudo_gradient_results) / len(pseudo_gradient_results)
        # weights_results = [
        #     (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
        #     for _, fit_res in results
        # ]

        

        # aggregated_weights = aggregate(weights_results)
        trained_layer = results[0][1].metrics["trained_layer"]
        # print(f"aggregated weight size {len(aggregated_weights)} ")

        if trained_layer == -1:
            update_direction = self.pseudo_gradient_update(parameters_to_ndarrays(self.latest_parameters),results)
            current_model = parameters_to_ndarrays(self.latest_parameters)
            current_flat = np.concatenate([param.flatten() for param in current_model])
            
            updated_model = [current_layer + update_layer for current_layer, update_layer in zip(current_model, update_direction)]
            updated_flat = np.concatenate([param.flatten() for param in updated_model])
            print(f"Number of parameters after update: {len(updated_flat)}")
            self.latest_parameters = ndarrays_to_parameters(updated_model)
        else:
            current_model = parameters_to_ndarrays(self.latest_parameters)
            update_direction = self.pseudo_gradient_update([current_model[trained_layer* 2], current_model[trained_layer* 2 + 1]],results)
            print(f"updateing layers {trained_layer* 2}  and {trained_layer* 2 + 1} ")
            current_model[trained_layer* 2] = np.add(current_model[trained_layer* 2], update_direction[0])
            current_model[trained_layer* 2 +1] = np.add(current_model[trained_layer* 2 +1], update_direction[1])
            self.latest_parameters = ndarrays_to_parameters(current_model)

        self.update_full_client_models(server_round, results, self.previous_parameters)

        self.update_metrics(server_round, self.latest_parameters, results)

        metrics_aggregated = {}
        return self.latest_parameters, metrics_aggregated

    

