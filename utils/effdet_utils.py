import cv2
import numpy as np
from PIL import Image
import torch
from effdet import create_model, DetBenchPredict

def load_model(model_path, device):
    model = create_model('tf_efficientdet_d0', bench_task='predict', num_classes=3, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = DetBenchPredict(model)
    model.eval().to(device)
    return model

def preprocess(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((512, 512))
    image_np = np.array(image) / 255.0
    image_tensor = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0).float()
    return image_tensor

def draw_boxes(original_path, boxes, scores, labels, save_path, threshold=0.3):
    image = cv2.imread(original_path)
    h, w = image.shape[:2]

    for box, score, label in zip(boxes, scores, labels):
        if score < threshold:
            continue
        x1, y1, x2, y2 = box.astype(int)
        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f'{label}: {score:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imwrite(save_path, image)

def predict_image(image_path, model, device, save_path='static/uploaded/result.jpg'):
    input_tensor = preprocess(image_path).to(device)
    with torch.no_grad():
        output = model(input_tensor)[0]

    boxes = output['boxes'].cpu().numpy()
    scores = output['scores'].cpu().numpy()
    labels = output['labels'].cpu().numpy()

    # Draw and save image with boxes
    draw_boxes(image_path, boxes, scores, labels, save_path)
    return save_path
