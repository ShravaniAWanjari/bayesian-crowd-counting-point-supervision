import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CrowdDataset
from model import CrowdCounterNet
from loss import BayesianLoss

# Define the checkpoint file name
CHECKPOINT_PATH = "checkpoint.pth"

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'Early Stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def save_checkpoint(model, optimizer, epoch, loss, path=CHECKPOINT_PATH):
    print(f"Saving checkpoint to {path}...")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print("Checkpoint saved.")

def load_checkpoint(model, optimizer, path=CHECKPOINT_PATH):
    if not os.path.exists(path):
        print("No checkpoint found. Starting from scratch.")
        return 0
    
    print(f"Loading checkpoint from {path}...")
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Checkpoint loaded. Resuming training from epoch {start_epoch}.")
    return start_epoch

def train_and_evaluate(root_dir, num_epochs=500, learning_rate=1e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Dataset and DataLoader Setup ---
    train_dataset = CrowdDataset(root_dir=root_dir, subset='Train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=os.cpu_count()
    )

    # --- Model, Loss, and Optimizer Instantiation ---
    model = CrowdCounterNet(pretrained=True).to(device)
    
    sigma_param = 8
    d_param = 0.15 * 512
    loss_fn = BayesianLoss(sigma=sigma_param, d=d_param).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    start_epoch = load_checkpoint(model, optimizer)

    # --- Main Training Loop with Early Stopping ---
    early_stopping = EarlyStopping(patience=10)
    best_mae = float('inf')

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, points in pbar:
            if images is None: continue
            images = images.to(device)
            points = points[0].to(device)
            optimizer.zero_grad()
            pred_density = model(images)
            loss = loss_fn(pred_density, points)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {avg_loss:.4f}")
        
        # --- Evaluation on Test Set ---
        mae, mse = evaluate_model(model, root_dir, subset='Test')
        
        # --- Early Stopping Logic ---
        early_stopping(mae)
        if mae < best_mae:
            best_mae = mae
            print(f"Test MAE improved to {best_mae:.2f}. Saving best model...")
            torch.save(model.state_dict(), 'best_model.pth')
            
        if early_stopping.early_stop:
            print("Early stopping triggered. Training finished.")
            break
            
        # Save a regular checkpoint at the end of each epoch
        save_checkpoint(model, optimizer, epoch, avg_loss)
            
    return best_mae, mse

def evaluate_model(model, root_dir, subset='Test'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_dataset = CrowdDataset(root_dir=root_dir, subset=subset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=os.cpu_count()
    )

    mae = 0.0
    mse = 0.0
    total_images = len(test_loader)

    print("Starting evaluation...")
    with torch.no_grad():
        for i, (images, points) in enumerate(tqdm(test_loader, desc="Evaluating")):
            if images is None: continue
            images = images.to(device)
            
            pred_density = model(images)
            
            estimated_count = torch.sum(pred_density).item()
            
            ground_truth_count = points[0].size(0)
            
            mae += abs(estimated_count - ground_truth_count)
            mse += (estimated_count - ground_truth_count)**2

    mae = mae / total_images
    mse = np.sqrt(mse / total_images)

    print(f"Evaluation complete. MAE: {mae:.2f}, MSE: {mse:.2f}")
    return mae, mse

if __name__ == '__main__':
    dataset_root = 'data/UCF-QNRF_ECCV18'
    
    best_mae, best_mse = train_and_evaluate(
        root_dir=dataset_root, 
        num_epochs=200
    )
    
    print(f"\nFinal Best Results: MAE: {best_mae:.2f}, MSE: {best_mse:.2f}")