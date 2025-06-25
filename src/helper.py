import sys
from flwr.common import Parameters
import base64
import pickle
import torch





def get_parameters_size(params: Parameters) -> int:
    size = sys.getsizeof(params)  
    size += sys.getsizeof(params.tensor_type)  
    size += sys.getsizeof(params.tensors)  
    size += sum(sys.getsizeof(tensor) for tensor in params.tensors) 
    return size



def serialize_optimizer_state(state_dict):
    """Serialize optimizer state with reduced memory footprint"""
    lightweight_state = {'state': {}, 'param_groups': state_dict['param_groups']}
    
    for k, v in state_dict['state'].items():
        lightweight_state['state'][k] = {
            'exp_avg': v['exp_avg'].clone().detach().cpu().half().numpy(),  # Use half precision
            'exp_avg_sq': v['exp_avg_sq'].clone().detach().cpu().half().numpy(),
            'step': v['step'].item()  # Store as scalar instead of tensor
        }
    
    return base64.b64encode(pickle.dumps(lightweight_state, protocol=pickle.HIGHEST_PROTOCOL)).decode('ascii')

def deserialize_optimizer_state(serialized_state):
    """Deserialize and reconstruct optimizer state"""
    state_dict = pickle.loads(base64.b64decode(serialized_state))
    reconstructed_state = {'state': {}, 'param_groups': state_dict['param_groups']}
    
    for k, v in state_dict['state'].items():
        reconstructed_state['state'][k] = {
            'exp_avg': torch.tensor(v['exp_avg'], dtype=torch.float32),
            'exp_avg_sq': torch.tensor(v['exp_avg_sq'], dtype=torch.float32),
            'step': torch.tensor(v['step'])
        }
    
    return reconstructed_state


