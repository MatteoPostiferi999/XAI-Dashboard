import torch
from PIL import Image
from torchvision import transforms

def predict(image: Image.Image, model, device, class_labels: list[str]) -> tuple[str, float, dict]:
    # Preprocessing
    image = image.convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        probs = torch.nn.functional.softmax(output[0], dim=0)

        confidence, predicted_idx = torch.max(probs, 0)
        predicted_class_name = class_labels[predicted_idx.item()]

        # Build probability dictionary
        prob_dict = {
            class_labels[i]: round(probs[i].item(), 4)
            for i in range(len(class_labels))
        }

    return predicted_class_name, confidence.item(), prob_dict
