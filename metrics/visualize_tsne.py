import argparse
from collections import OrderedDict
import contextlib
from copy import deepcopy
from datetime import datetime
import functools
import json
import logging
import multiprocessing as mp
import os
import socket
import subprocess
from time import time, sleep

from PIL import Image
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

from transformers import CLIPVisionModel, AutoProcessor
import torch.nn.functional as F
import imageio
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def evaluate_single(video_path, transform, feature_extractor):
    reader = imageio.get_reader(video_path)

    features = []

    with torch.no_grad():
        for i, image in enumerate(reader):
            image = transform(Image.fromarray(image)).unsqueeze(0).cuda()
            # Map input images to latent space + normalize latents:
            if args.feature_extractor == 'clip':
                feat = feature_extractor(image)['pooler_output']
            elif args.feature_extractor == 'dino':
                feat = feature_extractor(image)
            elif args.feature_extractor == 'resnet':
                feat = feature_extractor(image)
            else:
                raise NotImplementedError
            features.append(feat)
    
    features = torch.cat(features, dim=0)
    reader.close()

    features = features.detach().cpu().numpy()

    # 3. t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features)

    # 4. visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(features_tsne[:, 0], features_tsne[:, 1], alpha=0.5, s=200)
    plt.axis('off')
    # plt.title('t-SNE of Video Frame Features')
    # plt.xlabel('t-SNE Component 1')
    # plt.ylabel('t-SNE Component 2')
    plt.savefig(args.result_path, dpi=300, bbox_inches='tight')



def main(args):

    # Setup data:
    if args.feature_extractor == 'clip':
        transform = transforms.Compose([
            transforms.Lambda(
            functools.partial(center_crop_arr, image_size=args.image_size)
        ),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711],
                                 inplace=True),
        ])
    elif args.feature_extractor == 'dino':
        transform = transforms.Compose([
            transforms.Lambda(
            functools.partial(center_crop_arr, image_size=args.image_size)
        ),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225],
                                 inplace=True),
        ])
    elif args.feature_extractor == 'resnet':
        transform = transforms.Compose([
            transforms.Lambda(
            functools.partial(center_crop_arr, image_size=args.image_size)
        ),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225],
                                 inplace=True),
        ])
    else:
        raise NotImplementedError
    # dataset = ImageFolder(args.data_path, transform=transform)
    if args.feature_extractor == 'clip':
        feature_extractor = CLIPVisionModel.from_pretrained(args.extractor_path).cuda()
    elif args.feature_extractor == 'dino':
        feature_extractor = torch.hub.load(args.extractor_path.split(',')[0], args.extractor_path.split(',')[1]).cuda()
    elif args.feature_extractor == 'resnet':
        feature_extractor = torchvision.models.resnet50(pretrained=True).cuda()
        feature_extractor.fc = nn.Identity()
    else:
        raise NotImplementedError

    evaluate_single(args.video_path, transform, feature_extractor)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--result_path", type=str, default='tsne_plot.png')
    parser.add_argument("--feature_extractor", type=str,
                        default='clip',)
    parser.add_argument("--extractor_path", type=str,
                        # default="openai/clip-vit-large-patch14-336",
                        default='openai/clip-vit-base-patch32',)
    parser.add_argument("--feature_dimension", type=int,
                        default=768,)
    parser.add_argument("--image_size", type=int, 
                        default=224)
    args = parser.parse_args()

    main(args)