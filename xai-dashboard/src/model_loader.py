import torch
from torchvision import models

def load_model(model_path: str, num_classes: int, use_cuda: bool = True):
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    # Load ResNet18
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Move model to device and set to eval
    model.to(device)
    model.eval()

    return model, device
