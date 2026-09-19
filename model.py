"""Neural-network and Grad-CAM helpers shared by training and the web app."""

import cv2
import numpy as np
import torch
import torch.nn as nn


class CropDiseaseCNN(nn.Module):
    """Compact CNN suitable for CPU training in a hackathon prototype."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)
        self.target_layer = self.features[6]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.classifier(features.flatten(1))


def get_crop_model(num_classes: int) -> nn.Module:
    """Create the same lightweight model architecture for training and inference."""
    return CropDiseaseCNN(num_classes)


class GradCAM:
    """Grad-CAM for a single convolutional target layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_inputs, grad_outputs):
        self.gradients = grad_outputs[0].detach()

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: int):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, target_class].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        heatmap = cv2.resize(cam, (224, 224))
        return heatmap, output.detach()

    def close(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def apply_heatmap_overlay(image_rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Blend a Grad-CAM heatmap with an RGB leaf image."""
    heatmap_uint8 = np.uint8(255 * heatmap)
    coloured_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    coloured_heatmap = cv2.cvtColor(coloured_heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 0.55, coloured_heatmap, 0.45, 0)
