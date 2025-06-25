from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union, Callable
import pickle

import numpy as np
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

"""
Implementation of this file based on the previous project implementation but has been greatly modified and extneded
"""



class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 180)
        self.fc2 = nn.Linear(180, 160)
        self.fc3 = nn.Linear(160, 140)
        self.fc4 = nn.Linear(140, 120)
        self.fc5 = nn.Linear(120, 84)
        self.fc7 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc7(x)
        return x
    



DEVICE = "mps"
NETWORK_LEN = len(Net().state_dict().keys()) //2 
EPOCHS = 8
NUM_PARTITIONS = 6
NUM_OF_CYCLES  = 1
NUM_OF_FULL_UPDATES_BETWEEN_CYCLES = 2
NUM_OF_ROUNDS = (NUM_OF_CYCLES * NUM_OF_FULL_UPDATES_BETWEEN_CYCLES) + (NUM_OF_CYCLES * NETWORK_LEN *2)
GLOBAL_CRITERION = torch.nn.CrossEntropyLoss()
BACKEND_CONFIG = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}



def get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model: nn.Module, parameters: List[np.ndarray], trainable_layers=-1):
    current_state = OrderedDict(model.state_dict())
    
    if trainable_layers == -1:
        params_dict = zip(current_state.keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
    else:
        numpy_state = [param.cpu().numpy() for param in current_state.values()]
        numpy_state[trainable_layers*2] = parameters[0]
        numpy_state[trainable_layers*2 + 1] = parameters[1]
        for idx, key in enumerate(current_state.keys()):
            current_state[key] = torch.from_numpy(numpy_state[idx])
        
        model.load_state_dict(current_state, strict=True)

def fedAvg_train(model: nn.Module, train_loader: DataLoader, num_epochs: int):
    criterion = GLOBAL_CRITERION
    optimizer = torch.optim.Adam(model.parameters())

    model.train()
    for epoch in range(num_epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in train_loader:
            images, labels = batch["img"], batch["label"]
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(train_loader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")




def fedProx_train(model: nn.Module, train_loader: DataLoader, num_epochs: int, proximal_mu: float, global_params: List[torch.Tensor]):
    criterion = GLOBAL_CRITERION
    optimizer = torch.optim.Adam(model.parameters())

    model.train()
    for epoch in range(num_epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in train_loader:
            images, labels = batch["img"], batch["label"]
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            proximal_term = 0.0
            for local_weights, global_weights in zip(model.parameters(), global_params):
                proximal_term += (local_weights - global_weights).norm(2)
            loss = criterion(model(images), labels) + (proximal_mu / 2) * proximal_term
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(train_loader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")



def fedAvgMom_train(model: nn.Module, train_loader: DataLoader, num_epochs: int, optimizer=None):
    """Train the network on the training set."""
    print(f"training network...")
    criterion = torch.nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters()) 
    
    
    model.train()
    for epoch in range(num_epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in train_loader:
            images, labels = batch["img"], batch["label"]
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
                
            loss.backward()
            
            optimizer.step()
            epoch_loss += loss.item()  # Use .item() to avoid accumulating tensors
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        
        if len(train_loader.dataset) > 0:
            epoch_loss /= len(train_loader.dataset)
        if total > 0:
            epoch_acc = correct / total
        else:
            epoch_acc = 0.0
            
            
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

def fedProxMom_train(model: nn.Module, train_loader: DataLoader, num_epochs: int, proximal_mu:float, global_params:List[torch.Tensor], optimizer=None,):
    """Train the network on the training set."""
    print(f"training network...")
    criterion = torch.nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())
    model.train()
    for epoch in range(num_epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in train_loader:
            images, labels = batch["img"], batch["label"]
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            proximal_term = 0.0
            for local_weights, global_weights in zip(model.parameters(), global_params):
                proximal_term += (local_weights - global_weights).norm(2)
            loss = criterion(outputs, labels) + (proximal_mu / 2) * proximal_term

                
            loss.backward()
            
            
            optimizer.step()
            epoch_loss += loss.item()  # Use .item() to avoid accumulating tensors
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        
        if len(train_loader.dataset) > 0:
            epoch_loss /= len(train_loader.dataset)
        if total > 0:
            epoch_acc = correct / total
        else:
            epoch_acc = 0.0

            
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")



def test(model: nn.Module, test_loader: DataLoader) -> float:
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch["img"], batch["label"]
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(test_loader.dataset)
    accuracy = correct / total
    return loss, accuracy


#based on FedPart paper
def freeze_layers(model: torch.nn.Module, trainable_layers: int):
    trainable_indices = [-1] if trainable_layers == -1 else [trainable_layers * 2, trainable_layers * 2 + 1]
    
    for idx, (name, param) in enumerate(model.named_parameters()):
        is_trainable = idx in trainable_indices or trainable_indices[0] == -1
        param.requires_grad = is_trainable
        status = "trainable" if is_trainable else "frozen"
        print(f"Layer {idx} ({name}) is {status}")
        


def local_adam_train(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    lambda_: float = 1e-8,
    clip_rho: float = 1.0,
    initial_u: Optional[List[torch.Tensor]] = None,
    initial_v: Optional[List[torch.Tensor]] = None,
) -> Tuple[nn.Module, List[torch.Tensor], List[torch.Tensor]]:
    criterion = GLOBAL_CRITERION
    model.train()

    all_params = list(model.parameters())

    if initial_u is None:
        u = [torch.zeros_like(p.data) for p in all_params]
    else:
        u = []
        for i, p in enumerate(all_params):
            if i < len(initial_u) and initial_u[i].shape == p.data.shape:
                u.append(initial_u[i].clone().to(DEVICE))
            else:
                # Shape mismatch or missing - create zero tensor
                print(f"[DEBUG] Creating zero u tensor for param {i}, shape: {p.data.shape}")
                u.append(torch.zeros_like(p.data))

    if initial_v is None:
        v = [torch.zeros_like(p.data) for p in all_params]
    else:
        v = []
        for i, p in enumerate(all_params):
            if i < len(initial_v) and initial_v[i].shape == p.data.shape:
                v.append(initial_v[i].clone().to(DEVICE))
            else:
                # Shape mismatch or missing - create zero tensor
                print(f"[DEBUG] Creating zero v tensor for param {i}, shape: {p.data.shape}")
                v.append(torch.zeros_like(p.data))

    print(f"Starting explicit Local Adam training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in train_loader:
            images, labels = batch["img"], batch["label"]
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            model.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            with torch.no_grad():
                for i, p in enumerate(all_params):
                    if not p.requires_grad or p.grad is None:
                        continue

                    grad = p.grad.data
                    g_clipped = torch.sign(grad) * torch.min(torch.abs(grad), torch.tensor(clip_rho).to(DEVICE))
                    
                    u[i].mul_(beta1).add_(g_clipped, alpha=1 - beta1)
                    v[i].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    # --- CORRECTED UPDATE RULE ---
                    # Denominator is now sqrt(v + λ^2) as per the paper 
                    denominator = torch.sqrt(v[i] + lambda_**2)
                    
                    # Update model parameters (x) using the corrected formula 
                    p.data.addcdiv_(u[i], denominator, value=-lr)


            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_acc = correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}: train_loss {avg_epoch_loss:.5f}, accuracy {epoch_acc:.4f}")

    print("Local Adam training finished.")
    return model, u, v


def proximal_local_adam_train(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    lambda_: float = 1e-8,
    clip_rho: float = 1.0,
    initial_u: Optional[List[torch.Tensor]] = None,
    initial_v: Optional[List[torch.Tensor]] = None,
    proximal_mu: float = 0.0,
    global_params: Optional[List[torch.Tensor]] = None,
) -> Tuple[nn.Module, List[torch.Tensor], List[torch.Tensor]]:
    criterion = GLOBAL_CRITERION
    model.train()

    all_params = list(model.parameters())

    if initial_u is None:
        u = [torch.zeros_like(p.data) for p in all_params]
    else:
        u = []
        for i, p in enumerate(all_params):
            if i < len(initial_u) and initial_u[i].shape == p.data.shape:
                u.append(initial_u[i].clone().to(DEVICE))
            else:
                # Shape mismatch or missing - create zero tensor
                print(f"[DEBUG] Creating zero u tensor for param {i}, shape: {p.data.shape}")
                u.append(torch.zeros_like(p.data))

    if initial_v is None:
        v = [torch.zeros_like(p.data) for p in all_params]
    else:
        v = []
        for i, p in enumerate(all_params):
            if i < len(initial_v) and initial_v[i].shape == p.data.shape:
                v.append(initial_v[i].clone().to(DEVICE))
            else:
                # Shape mismatch or missing - create zero tensor
                print(f"[DEBUG] Creating zero v tensor for param {i}, shape: {p.data.shape}")
                v.append(torch.zeros_like(p.data))

    print(f"Starting explicit Local Adam training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in train_loader:
            images, labels = batch["img"], batch["label"]
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            model.zero_grad()
            outputs = model(images)
            proximal_term = 0.0
            for local_weights, global_weights in zip(model.parameters(), global_params):
                proximal_term += (local_weights - global_weights).norm(2)
            loss = criterion(outputs, labels) + (proximal_mu / 2) * proximal_term
            loss.backward()

            with torch.no_grad():
                for i, p in enumerate(all_params):
                    if not p.requires_grad or p.grad is None:
                        continue

                    grad = p.grad.data
                    g_clipped = torch.sign(grad) * torch.min(torch.abs(grad), torch.tensor(clip_rho).to(DEVICE))
                    
                    u[i].mul_(beta1).add_(g_clipped, alpha=1 - beta1)
                    v[i].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denominator = torch.sqrt(v[i] + lambda_**2)
                    p.data.addcdiv_(u[i], denominator, value=-lr)


            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_acc = correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}: train_loss {avg_epoch_loss:.5f}, accuracy {epoch_acc:.4f}")

    print("Local Adam training finished.")
    return model, u, v