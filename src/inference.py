import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import CrowdCounterNet
from loss import BayesianLoss

def run_inference_and_visualization(image_path, model_path, crop_size = 512):
    '''
    Loads the model.predicts and visualizes density map'''


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Loading model from : {model_path}')
    model = CrowdCounterNet(pretrained=False)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f'Error loading model state dict:{e}')

    model.to(device)
    model.eval()


    image = cv2.imread(image_path)
    if image is None:
        print(f'Error: could not read image at {image_path}')
        return

    original_h, original_w, _ = image.shape

    if original_h > crop_size and original_w> crop_size :
        image = cv2.resize(image,(crop_size, crop_size))
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_tensor = torch.from_numpy(image_rgb.copy()).permute(2,0,1).float()
    image_tensor = image_tensor / 255.0
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)


    with torch.no_grad():
        pred_density = model(image_tensor)

    pred_density = pred_density.squeeze().cpu().numpy()

    estimated_count = pred_density.sum()

    print(f'\nEstimated crowd count: {estimated_count}')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    ax1.imshow(image_rgb)
    ax1.set_title(f'Orignal Image\n Estimated Count: {estimated_count:.2f}')
    ax1.axis('off')

    density_map = ax2.imshow(pred_density, cmap='jet')
    ax2.set_title('Predicted Density Map')
    ax2.axis('off')

    fig.colorbar(density_map, ax=ax2, orientation='vertical')

    plt.show()

if __name__ == '__main__':
    inference_image_path = "data/UCF-QNRF_ECCV18/Test/img_0010.jpg"

    best_model_path = "best_model.pth"

    run_inference_and_visualization(inference_image_path, best_model_path)