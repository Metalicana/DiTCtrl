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

def uniformity_score(points):
    """
    Calculate uniformity score between adjacent points using dot product similarity
    
    Args:
        points: numpy array, shape (n_samples, n_dimensions)
    
    Returns:
        float: uniformity score between 0-1, closer to 1 means more uniform distribution
    """
    # Normalize points
    normalized_points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Calculate dot product similarity between adjacent points
    similarities = np.sum(normalized_points[:-1] * normalized_points[1:], axis=1)
    
    # Calculate coefficient of variation CV = σ/μ * 10
    cv = np.std(similarities) / np.mean(similarities) * 10
    
    score = 1/(1 + cv)
    
    return score

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
    
    score = uniformity_score(features)
    return score


def main(args):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    results_file = f"evaluation_results_cscv/cscv_{timestamp}.json"
    os.makedirs("evaluation_results_cscv", exist_ok=True)
    
    results = {
        "new_metric": {
            "average_score": 0.0,
            "details": []
        }
    }
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

    all_frames_score = []
    
    target_seed_path = f"seed_{args.target_seed}"
    for root, dirs, files in os.walk(args.video_path):
        if target_seed_path not in root:
            continue
        for file in files:
            if file.lower().endswith('.mp4'):
                file_path = os.path.join(root, file)
                print(f"found MP4 file: {file_path}")
                score = evaluate_single(file_path, transform, feature_extractor)
                all_frames_score.append(score)
                
                results["new_metric"]["details"].append({
                    "video_path": file_path,
                    "video_results": score
                })
    
    average_score = sum(all_frames_score) / (len(all_frames_score) + 1e-5)
    results["new_metric"]["average_score"] = average_score
    
    # save results to JSON
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print("file_path: ", args.video_path)
    print("length: ", len(all_frames_score))
    print("target_seed: ", args.target_seed)
    print("score: ", average_score)
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--target_seed", type=int, required=True,
                        help="one seed path of results")
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
