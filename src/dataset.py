import os
import cv2
import scipy.io
import torch
import random
import numpy as np
import glob

class CrowdDataset:
        def __init__(self, root_dir, subset):

            self.root_dir = root_dir
            self.subset = subset
            self.image_files = []
            self.annotation_files = []

            data_path = os.path.join(self.root_dir, self.subset)

            image_paths = glob.glob(os.path.join(data_path, '*.jpg'))
            image_paths.sort()

            for img_path in image_paths:
                ann_path = img_path.replace('.jpg', '_ann.mat')

                if os.path.exists(ann_path):
                    self.image_files.append(img_path)
                    self.annotation_files.append(ann_path)
                else:
                    print(f"Warning: Annotation file not found for {img_path}")

            print(f"Found {len(self.image_files)} image-annotation pairs in the '{self.subset}' subset.")

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):

            try:
                image_path = self.image_files[idx]
                annotation_path = self.annotation_files[idx]

                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Could not read image at {image_path}. Skipping.")
                    return None, None 

                mat_data = scipy.io.loadmat(annotation_path)
                points = mat_data['annPoints'].astype(float)
            
                h, w, _ = image.shape
                crop_size = 512
            
                if h >= crop_size and w >= crop_size:
                    h_start = random.randint(0, h - crop_size)
                    w_start = random.randint(0, w - crop_size)
                
                    image = image[h_start:h_start + crop_size, w_start:w_start + crop_size]
                
                    mask = (points[:, 0] >= w_start) & (points[:, 0] < w_start + crop_size) & \
                        (points[:, 1] >= h_start) & (points[:, 1] < h_start + crop_size)
                    points = points[mask]
                
                    points[:, 0] -= w_start
                    points[:, 1] -= h_start

                if random.random() > 0.5:
                    image = cv2.flip(image, 1)
                    w = image.shape[1]
                    points[:, 0] = w - points[:, 0]


                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                image = torch.from_numpy(image.copy()).permute(2, 0, 1).float()
            
                image = image / 255.0

                points = torch.from_numpy(points.copy()).float()
            
                return image, points
            
            except Exception as e:
                print(f"Error processing image {self.image_files[idx]}: {e}. Skipping.")
                return None, None