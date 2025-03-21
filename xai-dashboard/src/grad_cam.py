import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_gradcam(image_tensor, model, target_layer):
    gradients = []
    feature_maps = []

    # Set up hooks to capture the feature maps and gradients
    def forward_hook(module, input, output):
        feature_maps.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])  # Capture the gradients

    # Register the hooks to the target layer
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    # Perform the forward pass
    output = model(image_tensor)
    _, predicted_class = torch.max(output, 1)

    # Perform the backward pass (gradient calculation)
    model.zero_grad()  # Zero out previous gradients
    output[0, predicted_class].backward()  # Backpropagate using the predicted class

    # Get the gradients and feature maps
    gradient = gradients[0]
    feature_map = feature_maps[0]

    # Weight the feature maps by the gradients
    weights = torch.mean(gradient, dim=(2, 3), keepdim=True)  # Global average pooling
    weighted_feature_map = weights * feature_map

    # Sum the weighted feature maps and apply ReLU
    grad_cam = torch.sum(weighted_feature_map, dim=1).squeeze()
    grad_cam = np.maximum(grad_cam.cpu().data.numpy(), 0)  # Apply ReLU

    # Resize the heatmap to match the input image
    grad_cam = cv2.resize(grad_cam, (image_tensor.shape[2], image_tensor.shape[3]))
    grad_cam = np.uint8(255 * grad_cam / np.max(grad_cam))  # Normalize to [0, 255]
    heatmap = cv2.applyColorMap(grad_cam, cv2.COLORMAP_JET)

    # Convert the input image to numpy and overlay the heatmap
    image = image_tensor.squeeze().cpu().data.numpy().transpose(1, 2, 0)
    heatmap = np.float32(heatmap) / 255
    image = np.float32(image)
    superimposed_image = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)

    return superimposed_image
