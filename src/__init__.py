import os
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