import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np

from dataset import CrowdDataset
from model import CrowdCounterNet
from loss import BayesianLoss

def train_model(root_dir, num_epochs=100, learning_rate=1e-5):

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    BATCH_SIZE = 1

    train_dataset = CrowdDataset(root_dir = root_dir, subset='Train')
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=os.cpu_count()
    )

    model =  CrowdCounterNet(pretrained=True).to(device)

    loss_fn = BayesianLoss().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for i, (image, points) in enumerate(train_loader):
            images = images.to(device)
            points = points[0].to(device)

            optimizer.zero_grad()

            pred_density = model(images)

            loss = loss_fn(pred_density, points)

            loss.backwards()

            optimizer.step()

            epoch_loss += loss.item()

            pbar.set_postfix(loss = loss.item())
        avg_loss = epoch_loss/len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

    print("Training Finished")
    return model

def evaluate_model(model, root_dir, subset='Test'):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    test_dataset = CrowdDataset(root_dir=root_dir, subset=subset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle = False,
        num_workers = os.cpu_count()
    )

    mae = 0.0
    mse = 0.0
    total_images = len(test_loader)

    print("Starting evaluation...")
    with torch.no_grad():
        for i, (images, points) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)
            pred_density = model(images)

            estimated_count = torch.sum(pred_density).item()

            ground_truth_count = points[0].size(0)

            mae += abs(estimated_count - ground_truth_count)
            mse += (estimated_count - ground_truth_count)**2

        mae = mae / total_images
        mse = np.sqrt(mse/total_images)

        print(f"Evaluation complete. MAE : {mae:.2f}, MSE:{mse:.2f}")
        return mae, mse

if __name__ == '__main__':

    dataset_root = 'data/UCF-QNRF_ECCV18'
    trained_model = train_model(root_dir='data/UCF-QNRF_ECCV18', num_epochs=10)
    evaluate_model(trained_model, root_dir=dataset_root)