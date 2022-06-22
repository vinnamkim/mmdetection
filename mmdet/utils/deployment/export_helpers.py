from typing import List
import torch


def get_saliency_map(feature_map: torch.Tensor):
    saliency_map = torch.sigmoid(torch.mean(feature_map, dim=1))
    saliency_map = 255 * (saliency_map - torch.min(saliency_map))/(torch.max(saliency_map) - torch.min(saliency_map) + 1e-12)
    saliency_map = saliency_map.to(torch.uint8)
    return saliency_map


def get_feature_vector(feature_maps: List[torch.Tensor]) -> torch.Tensor:
    pooled_features = [torch.nn.functional.adaptive_avg_pool2d(feat_map, (1, 1)) for feat_map in feature_maps]  # x - backbone+neck output
    feature_vector = torch.cat(pooled_features, dim=1)
    return feature_vector