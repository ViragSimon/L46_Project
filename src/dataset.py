from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from flwr_datasets.partitioner import DirichletPartitioner

"""
Implementation of this file based on the previous project implementation but has been greatly modified and extneded
"""

def load_datasets(partition_id, num_partitions: int):
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": num_partitions})
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    train_loader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    val_loader = DataLoader(partition_train_test["test"], batch_size=32)
    test_set = fds.load_split("test").with_transform(apply_transforms)
    test_loader = DataLoader(test_set, batch_size=32)
    return train_loader, val_loader, test_loader



def load_heterogenous_datasets(partition_id, num_partitions: int):
    drichlet_partitioner = DirichletPartitioner(num_partitions=num_partitions, alpha=0.1, partition_by="label")
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": drichlet_partitioner})
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    valloader = DataLoader(partition_train_test["test"], batch_size=32)
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=32)
    return trainloader, valloader, testloader