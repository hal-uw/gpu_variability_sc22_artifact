import os
import kfac
import torch
import torch.distributed as dist

#Testing lift of ImageNetV2 Extract
import pathlib
import tarfile
import requests
import shutil
from tqdm import tqdm
from PIL import Image

# from imagenetv2_pytorch import ImageNetV2Dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

URLS = {"matched-frequency" : "https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenetv2-matched-frequency.tar.gz",
        "threshold-0.7" : "https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenetv2-threshold0.7.tar.gz",
        "top-images": "https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenetv2-top-images.tar.gz",
        "val": "https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenet_validation.tar.gz"}

FNAMES = {"matched-frequency" : "imagenetv2-matched-frequency-format-val",
        "threshold-0.7" : "imagenetv2-threshold0.7-format-val",
        "top-images": "imagenetv2-top-images-format-val",
        "val": "imagenet_validation"}

V2_DATASET_SIZE = 10000
def load_imagenet(variant="matched-frequency", location="."):
#    dataset_root = pathlib.Path(f"{location}/ImageNetV2-{variant}/") #./ImageNetV2-matched-frequency/
    dataset_root = pathlib.Path(f"{location}/{FNAMES[variant]}/") #./imagenetv2-matched-frequency-format-val
    tar_root = pathlib.Path(f"{location}/ImageNetV2-{variant}.tar.gz")# ./ImageNetV2-matched-frequency.tar.gz/
    fnames = list(dataset_root.glob("**/*.jpeg"))
    assert variant in URLS, f"unknown V2 Variant: {variant}"
    if not dataset_root.exists() or len(fnames) != V2_DATASET_SIZE:
        if not tar_root.exists():
            print(f"Dataset {variant} not found on disk, downloading....")
            response = requests.get(URLS[variant], stream=True)
            total_size_in_bytes= int(response.headers.get('content-length', 0))
            block_size = 1024 #1 Kibibyte
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            with open(tar_root, 'wb') as f:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    f.write(data)
            progress_bar.close()
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                assert False, f"Downloading from {URLS[variant]} failed"
        print("Extracting....")
        tarfile.open(tar_root).extractall(f"{location}")# tar file gets extracted to ./imagenetv2-matched-frequency-format-val
        # shutil.move(f"{location}/{FNAMES[variant]}", dataset_root) #Rename from extracted dir to ImageNetV2-matched-frequency
    else:
        print("Extracted dataset already exists. Skipping download/extract of dataset.")

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
    #train_dataset = ImageNetV2Dataset("matched-frequency") # supports matched-frequency, threshold-0.7, top-images variants 
    load_imagenet()
    formatted_dataset = datasets.ImageFolder(
                "./imagenetv2-matched-frequency-format-val",
                transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]))
    return make_sampler_and_loader(args, formatted_dataset)

def make_sampler_and_loader(args, train_dataset):
    torch.set_num_threads(4)
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size * args.batches_per_allreduce,
            sampler=train_sampler, **kwargs)

    return train_sampler, train_loader

