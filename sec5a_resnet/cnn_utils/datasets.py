import os
import kfac
import torch
import torch.distributed as dist

from imagenetv2_pytorch import ImageNetV2Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_cifar(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    os.makedirs(args.data_dir, exist_ok=True)

    download = True if args.local_rank == 0 else False
    if not download: dist.barrier()
    train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, 
                                     download=download, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=args.data_dir, train=False,
                                    download=download, transform=transform_test)
    if download: dist.barrier()
    
    return make_sampler_and_loader(args, train_dataset, test_dataset)


def get_imagenet(args): 
    train_dataset = ImageNetV2Dataset("matched-frequency") # supports matched-frequency, threshold-0.7, top-images variants
    return make_sampler_and_loader(args, train_dataset)

def make_sampler_and_loader(args, train_dataset, val_dataset):
    torch.set_num_threads(4)
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size * args.batches_per_allreduce,
            sampler=train_sampler, **kwargs)

    return train_sampler, train_loader
