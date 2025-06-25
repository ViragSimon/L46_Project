
import matplotlib.pyplot as plt
import numpy as np
import os
markers = ['o', 's', '^', 'D', 'v', '*', 'p', 'h', '8', '+', 'x']
colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', '#FFA500', '#800080', '#008080', '#FF1493']

def plot_accuracy_per_round(all_results):
    plt.figure(figsize=(12, 5)) 
    
   
    i = 0
    for key, value in all_results.items():
        model_results = value[1]
        rounds = list(model_results.keys())
        accuracies = [model_results[round]["global_metrics"]["accuracy"] for round in rounds]
        
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        i += 1
        plt.plot(rounds, accuracies, 
                marker=marker, 
                linestyle='-', 
                color=color,
                label=key)

    plt.xlabel('Round', fontdict={'family': 'serif', 'weight': 'normal'})
    plt.xticks(range(0, max(rounds) + 1))
    plt.ylabel('Accuracy')
    plt.title('Global Model Accuracy for Each Round')
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
    plt.tight_layout()  
    plt.grid(True)

def plot_loss_per_round(all_results, modified_label:str = None):
    plt.figure(figsize=(12, 5))  

    i = 0
    for key, value in all_results.items():
        model_results = value[1]
        rounds = list(model_results.keys())
        accuracies = [model_results[round]["global_loss"] for round in rounds]
        
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        i += 1
        plt.plot(rounds, accuracies, 
                marker=marker, 
                linestyle='-', 
                color=color,
                label=key)

    plt.xlabel('Round', fontdict={'family': 'serif', 'weight': 'normal'})
    plt.xticks(range(0, max(rounds) + 1))
    plt.ylabel('Global Model Loss')
    if modified_label:
        plt.title(f'Global Model Loss for Each Round - {modified_label}')
    else:
        plt.title('Global Model Loss for Each Round')
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
    plt.tight_layout()
    plt.grid(True)

def plot_communication_cost_per_round(all_results, show_strategy_names:bool = False):
    plt.figure(figsize=(12, 5)) 
    
    i = 0
    for key, value in all_results.items():
        model_results = value[0]
        rounds = list(model_results.keys())
        accuracies = [model_results[round]["total_size"] for round in rounds]
        if show_strategy_names:
            label = strategy_names[key]
        else:
            label = key
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        i += 1
        plt.plot(rounds, accuracies, 
                marker=marker, 
                linestyle='-', 
                color=color,
                label=label)

    plt.xlabel('Round', fontdict={'family': 'serif', 'weight': 'normal'})
    plt.xticks(range(0, max(rounds) + 1))
    plt.ylabel('Communication Cost (bytes)')
    plt.title('Communication Cost for Each Round')
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
    plt.tight_layout()
    plt.grid(True)


strategy_names = {
    "fedavg": "FedAvg",
    "fedpartavg": "FedPartAvg",
    "fedavg_mom1_original": "FedAvgMom1",
    "fedavg_mom2_original": "FedAvgMom2",
    "fedavg_mom1_sigmoid": "FedAvgMom1",
    "fedavg_mom2_sigmoid": "FedAvgMom2",
    "fedpart_local_adam_avg": "Local Adam",
    "fedpseudo_gradient_avg_part": "Average Pseudo Gradient Update",
    "psap_absolute": "PSAP",
    "psap_sigmoid": "PSAP",
    "psap_original": "PSAP",
}

def plot_total_communication_cost_bar_chart(all_results):
    plt.figure(figsize=(12, 6)) 
    total_communication_costs = {}
    for key, value in all_results.items():
        model_results = value[0]
        key_name = strategy_names[key]
       
        total_communication_costs[key_name] = sum([model_results[round]["total_size"] for round in model_results])

    plt.bar(total_communication_costs.keys(), total_communication_costs.values())
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Training Strategy')
    plt.ylabel('Total Communication Cost (bytes)')
    plt.title('Total Communication Cost for Each Training Strategy')
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
    plt.tight_layout()
    plt.grid(True)

def plot_l2_norm_per_round(all_results, modified_label:str = None):
    plt.figure(figsize=(12, 5))  
    
    i = 0
    for key, value in all_results.items():
        metrics_history = value[2]
        rounds = metrics_history['rounds']
        l2_norms = metrics_history['l2_norms']
        
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        i += 1
        plt.plot(rounds, l2_norms,
                marker=marker,
                linestyle='-',
                color=color, 
                label=key)

    plt.xlabel('Round', fontdict={'family': 'serif', 'weight': 'normal'})
    plt.xticks(range(0, max(rounds) + 1))
    plt.ylabel('L2 Norm')   
    if modified_label:
        plt.title(f'L2 Norm per Round - {modified_label}')
    else:
        plt.title('L2 Norm per Round')
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
    plt.tight_layout()
    plt.grid(True)

def plot_parameter_difference_per_round(all_results, modified_label:str = None):
    plt.figure(figsize=(12, 5)) 
    
    i = 0
    for key, value in all_results.items():
        metrics_history = value[2]
        rounds = metrics_history['rounds']
        parameter_differences = metrics_history['parameter_differences']
        
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        i += 1
        plt.plot(rounds, parameter_differences,
                marker=marker,
                linestyle='-',
                color=color,
                label=key)

    plt.xlabel('Round', fontdict={'family': 'serif', 'weight': 'normal'})
    plt.xticks(range(0, max(rounds) + 1))
    plt.ylabel('Cumulative Parameter Difference From Previous Model')
    if modified_label:
        plt.title(f'Cumulative Parameter Difference per Round - {modified_label}')
    else:
        plt.title('Cumulative Parameter Difference per Round')
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
    plt.tight_layout()
    plt.grid(True)

def plot_cosine_similarity_per_round(all_results):
    plt.figure(figsize=(12, 5))  
    
    i = 0
    for key, value in all_results.items():
        metrics_history = value[2]
        rounds = metrics_history['rounds']
        cosine_similarities = metrics_history['cosine_similarities']
        
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        i += 1
        plt.plot(rounds, cosine_similarities,
                marker=marker,
                linestyle='-',
                color=color,
                label=key)

    plt.xlabel('Round', fontdict={'family': 'serif', 'weight': 'normal'})
    plt.xticks(range(0, max(rounds) + 1))
    plt.ylabel('Cosine Similarity to Previous Model')
    plt.title('Cosine Similarity Between Consecutive Central Models per Round')
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
    plt.tight_layout()
    plt.grid(True)

def plot_client_divergence_per_round(all_results):
    plt.figure(figsize=(12, 5))  
    
    i = 0
    for key, value in all_results.items():
        metrics_history = value[2]
        rounds = metrics_history['rounds']
        client_divergence = metrics_history['client_divergence']
        
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        i += 1
        plt.plot(rounds, client_divergence,
                marker=marker,
                linestyle='-',
                color=color,
                label=key)

    plt.xlabel('Round', fontdict={'family': 'serif', 'weight': 'normal'})
    plt.xticks(range(0, max(rounds) + 1))
    plt.ylabel('Client Divergence')
    plt.title('Client Divergence per Round')
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
    plt.tight_layout()
    plt.grid(True)

def plot_client_to_central_similarities_per_round(strategy_results):
    plt.figure(figsize=(12, 5)) 

    metrics_history = strategy_results[2]
    rounds = metrics_history['rounds']
    client_similarities = metrics_history['client_to_central_similarities']
    strategy_name = strategy_results[0]
    

    num_clients = len(client_similarities[0]) 
    
    
    for client_idx in range(num_clients):
        client_data = [round_similarities[client_idx] for round_similarities in client_similarities]
        
        marker = markers[client_idx % len(markers)]
        color = colors[client_idx % len(colors)]
        
        plt.plot(rounds, client_data,
                marker=marker,
                linestyle='-', 
                color=color,
                label=f'Client {client_idx}')

    plt.xlabel('Round', fontdict={'family': 'serif', 'weight': 'normal'})
    plt.xticks(range(0, max(rounds) + 1))
    plt.ylabel('Client to Central Similarity')
    plt.title(f'Client to Central Similarities per Round - {strategy_name}')
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
    plt.tight_layout()
    plt.grid(True)


def visualize_data(data):
    plot_accuracy_per_round(data)
    plot_loss_per_round(data)
    plot_communication_cost_per_round(data)
    plot_total_communication_cost_bar_chart(data)
    plot_l2_norm_per_round(data)
    plot_parameter_difference_per_round(data)
    plot_cosine_similarity_per_round(data)
    plot_client_divergence_per_round(data)
