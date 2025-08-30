# Original code inspired by https://github.com/deepinsight/insightface
# Modified for LFW dataset evaluation
import argparse
import os
import sys
import timeit
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import normalize
import prettytable

# Add parent directory to path to import onnx_helper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onnx_helper import ArcFaceORT

# Standard face alignment landmarks for 112x112 images
SRC = np.array([
    [30.2946, 51.6963],
    [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.2041]
], dtype=np.float32)
SRC[:, 0] += 8.0


class LFWDataset(Dataset):
    """LFW Dataset for face verification evaluation"""
    
    def __init__(self, data_root, pairs_file, align=True, image_size=(112, 112)):
        """
        Args:
            data_root: Path to lfw-deepfunneled directory
            pairs_file: Path to pairs CSV file
            align: Whether to apply face alignment (if landmarks available)
            image_size: Target image size for face recognition model
        """
        self.data_root = data_root
        self.image_size = image_size
        self.align = align
        
        # Load pairs from CSV
        self.pairs_df = pd.read_csv(pairs_file)
        self.pairs = []
        self.labels = []
        
        # Process pairs - the CSV contains both same-person and different-person pairs
        for _, row in self.pairs_df.iterrows():
            # Check if this is a same-person pair or different-person pair
            # Same person: name,imagenum1,imagenum2,
            # Different person: name1,imagenum1,name2,imagenum2
            
            if len(row) >= 4 and pd.notna(row.iloc[3]) and str(row.iloc[3]).strip() != '':
                # Different person pair (4 columns)
                name1 = row.iloc[0]
                img1_num = int(row.iloc[1])
                name2 = row.iloc[2]
                img2_num = int(row.iloc[3])
                
                img1_path = os.path.join(data_root, name1, f"{name1}_{img1_num:04d}.jpg")
                img2_path = os.path.join(data_root, name2, f"{name2}_{img2_num:04d}.jpg")
                
                if os.path.exists(img1_path) and os.path.exists(img2_path):
                    self.pairs.append((img1_path, img2_path))
                    self.labels.append(0)  # Different person
            else:
                # Same person pair (3 columns + empty 4th)
                name = row.iloc[0]
                img1_num = int(row.iloc[1])
                img2_num = int(row.iloc[2])
                
                img1_path = os.path.join(data_root, name, f"{name}_{img1_num:04d}.jpg")
                img2_path = os.path.join(data_root, name, f"{name}_{img2_num:04d}.jpg")
                
                if os.path.exists(img1_path) and os.path.exists(img2_path):
                    self.pairs.append((img1_path, img2_path))
                    self.labels.append(1)  # Same person
        
        # Add mismatched pairs if available
        mismatch_files = [
            'mismatchpairsDevTest.csv',
            'mismatchpairsDevTrain.csv'
        ]
        
        for mismatch_file in mismatch_files:
            mismatch_path = os.path.join(os.path.dirname(pairs_file), mismatch_file)
            if os.path.exists(mismatch_path):
                mismatch_df = pd.read_csv(mismatch_path)
                for _, row in mismatch_df.iterrows():
                    # Format: name,imagenum1,name,imagenum2 (different persons)
                    if len(row) == 4:
                        name1 = row.iloc[0]
                        img1_num = int(row.iloc[1])
                        name2 = row.iloc[2]
                        img2_num = int(row.iloc[3])
                        
                        img1_path = os.path.join(data_root, name1, f"{name1}_{img1_num:04d}.jpg")
                        img2_path = os.path.join(data_root, name2, f"{name2}_{img2_num:04d}.jpg")
                        
                        if os.path.exists(img1_path) and os.path.exists(img2_path):
                            self.pairs.append((img1_path, img2_path))
                            self.labels.append(0)  # Different person
            
        # Add matched pairs if available
        match_files = [
            'matchpairsDevTest.csv',
            'matchpairsDevTrain.csv'
        ]    
        
        for match_file in match_files:
            match_path = os.path.join(os.path.dirname(pairs_file), match_file)
            if os.path.exists(match_path):
                match_df = pd.read_csv(match_path)
                # Format: name,imagenum1,imagenum2 (same person)
                for _, row in match_df.iterrows():
                    if len(row) == 3:
                        name = row.iloc[0]
                        img1_num = int(row.iloc[1])
                        img2_num = int(row.iloc[2])
                        img1_path = os.path.join(data_root, name, f"{name}_{img1_num:04d}.jpg")
                        img2_path = os.path.join(data_root, name, f"{name}_{img2_num:04d}.jpg")
                        
                        if os.path.exists(img1_path) and os.path.exists(img2_path):
                            self.pairs.append((img1_path, img2_path))
                            self.labels.append(1)  # Same person
                       
        print(f"Loaded {len(self.pairs)} pairs ({sum(self.labels)} positive, {len(self.labels) - sum(self.labels)} negative)")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img1_path, img2_path = self.pairs[idx]
        label = self.labels[idx]
        
        # Load and preprocess images
        img1 = self._load_and_preprocess_image(img1_path)
        img2 = self._load_and_preprocess_image(img2_path)
        
        return img1, img2, label
    
    def _load_and_preprocess_image(self, img_path):
        """Load and preprocess a single image"""
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        img = cv2.resize(img, self.image_size)
        
        # Convert to torch tensor
        img = img.astype(np.float32)
        
        # Convert to CHW format
        img = np.transpose(img, (2, 0, 1))
        
        return torch.from_numpy(img)


class LFWEvaluator:
    """LFW Face Verification Evaluator"""
    
    def __init__(self, model_path, use_flip=True):
        """
        Args:
            model_path: Path to ONNX model directory
            use_flip: Whether to use flip test augmentation
        """
        self.model = ArcFaceORT(model_path)
        self.model.check()
        self.use_flip = use_flip
        
        print(f"Model loaded: {model_path}")
        print(f"Feature dimension: {self.model.feat_dim}")
        print(f"Input size: {self.model.image_size}")
    
    def extract_features(self, dataloader):
        """Extract features from all image pairs"""
        all_features1 = []
        all_features2 = []
        all_labels = []
        
        print("Extracting features...")
        for batch_idx, (img1_batch, img2_batch, labels) in enumerate(dataloader):
            # Process image 1
            feat1 = self._extract_batch_features(img1_batch)
            feat2 = self._extract_batch_features(img2_batch)
            
            all_features1.append(feat1)
            all_features2.append(feat2)
            all_labels.extend(labels.numpy())
            
            if batch_idx % 100 == 0:
                print(f"Processed {batch_idx * dataloader.batch_size} pairs")
        
        # Concatenate all features
        features1 = np.vstack(all_features1)
        features2 = np.vstack(all_features2)
        labels = np.array(all_labels)
        
        print(f"Extracted features for {len(labels)} pairs")
        return features1, features2, labels
    
    def _extract_batch_features(self, img_batch):
        """Extract features from a batch of images"""
        batch_size = img_batch.shape[0]
        
        # Convert to numpy and adjust format for ONNX model
        imgs = img_batch.numpy()
        
        # Apply model normalization
        imgs = (imgs - self.model.input_mean) / self.model.input_std
        
        features = []
        
        for i in range(batch_size):
            img = imgs[i:i+1]  # Keep batch dimension
            
            if self.use_flip:
                # Original image
                feat1 = self.model.session.run(self.model.output_names, {self.model.input_name: img})[0]
                
                # Flipped image
                img_flipped = img[:, :, :, ::-1].copy()  # Flip horizontally
                feat2 = self.model.session.run(self.model.output_names, {self.model.input_name: img_flipped})[0]
                
                # Average features
                feat = (feat1 + feat2) / 2.0
            else:
                feat = self.model.session.run(self.model.output_names, {self.model.input_name: img})[0]
            
            features.append(feat[0])  # Remove batch dimension
        
        return np.array(features)
    
    def evaluate(self, features1, features2, labels):
        """Evaluate face verification performance"""
        # Normalize features
        features1 = normalize(features1, axis=1)
        features2 = normalize(features2, axis=1)
        
        # Compute cosine similarity
        similarities = np.sum(features1 * features2, axis=1)
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(labels, similarities)
        auc_score = auc(fpr, tpr)
        
        # Find best threshold (equal error rate)
        eer_threshold = thresholds[np.argmin(np.abs(tpr - (1 - fpr)))]
        eer = fpr[np.argmin(np.abs(tpr - (1 - fpr)))]
        
        # Compute accuracy at best threshold
        predictions = similarities > eer_threshold
        accuracy = np.mean(predictions == labels)
        
        # Compute TAR (True Accept Rate) at specific FAR (False Accept Rate) levels
        far_levels = [1e-4, 1e-3, 1e-2, 1e-1]
        tar_at_far = []
        
        for far in far_levels:
            # Find threshold that gives approximately this FAR
            if np.any(fpr <= far):
                threshold_idx = np.where(fpr <= far)[0][-1]
                tar = tpr[threshold_idx]
            else:
                tar = 0.0
            tar_at_far.append(tar)
        
        results = {
            'auc': auc_score,
            'eer': eer,
            'accuracy': accuracy,
            'best_threshold': eer_threshold,
            'tar_at_far': dict(zip(far_levels, tar_at_far))
        }
        
        return results, similarities, labels
    
    def print_results(self, results):
        """Print evaluation results in a formatted table"""
        print("\n" + "="*60)
        print("LFW Face Verification Results")
        print("="*60)
        
        # Main metrics
        table = prettytable.PrettyTable(['Metric', 'Value'])
        table.add_row(['AUC', f'{results["auc"]:.4f}'])
        table.add_row(['EER', f'{results["eer"]:.4f}'])
        table.add_row(['Accuracy', f'{results["accuracy"]:.4f}'])
        table.add_row(['Best Threshold', f'{results["best_threshold"]:.4f}'])
        
        print(table)
        
        # TAR at FAR levels
        print(f"\nTAR (True Accept Rate) at various FAR (False Accept Rate) levels:")
        tar_table = prettytable.PrettyTable(['FAR', 'TAR'])
        for far, tar in results['tar_at_far'].items():
            tar_table.add_row([f'{far:.0e}', f'{tar:.4f}'])
        
        print(tar_table)


def main(args):   
    start_time = timeit.default_timer()
    
    dataset = LFWDataset(
        data_root=args.data_root,
        pairs_file=args.pairs_file,
        align=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    load_time = timeit.default_timer() - start_time
    print(f"Dataset loaded in {load_time:.2f}s")
    
    evaluator = LFWEvaluator(args.model_path, use_flip=args.use_flip)
    
    # Extract features
    start_time = timeit.default_timer()
    features1, features2, labels = evaluator.extract_features(dataloader)
    extract_time = timeit.default_timer() - start_time
    print(f"Feature extraction completed in {extract_time:.2f}s")
    
    # Evaluate
    start_time = timeit.default_timer()
    results, similarities, labels = evaluator.evaluate(features1, features2, labels)
    eval_time = timeit.default_timer() - start_time
    print(f"Evaluation completed in {eval_time:.2f}s")
    
    # Print results
    evaluator.print_results(results)
    
    # Save results
    if args.save_results:
        save_data = {
            'results': results,
            'similarities': similarities,
            'labels': labels,
            'features1': features1,
            'features2': features2
        }
        np.save(args.save_results, save_data)
        print(f"Results saved to: {args.save_results}")
    
    print(f"\nTotal evaluation time: {load_time + extract_time + eval_time:.2f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LFW Face Verification Evaluation')
    
    # Model arguments
    parser.add_argument('--model-path', required=True, type=str,
                        help='Path to ONNX model directory')
    
    # Data arguments
    parser.add_argument('--data-root', default='./data/lfw/lfw-deepfunneled/lfw-deepfunneled', type=str,
                        help='Path to LFW dataset root directory')
    parser.add_argument('--pairs-file', default='./data/lfw/pairs.csv', type=str,
                        help='Path to pairs CSV file')
    
    # Evaluation arguments
    parser.add_argument('--batch-size', default=32, type=int,
                        help='Batch size for feature extraction')
    parser.add_argument('--num-workers', default=4, type=int,
                        help='Number of data loading workers')
    parser.add_argument('--use-flip', action='store_true', default=True,
                        help='Use flip test augmentation')
    parser.add_argument('--no-flip', dest='use_flip', action='store_false',
                        help='Disable flip test augmentation')
    
    # Output arguments
    parser.add_argument('--save-results', type=str, default='./results/results.npy',
                        help='Path to save detailed results (optional)')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model path does not exist: {args.model_path}")
    if not os.path.exists(args.data_root):
        raise ValueError(f"Data root does not exist: {args.data_root}")
    if not os.path.exists(args.pairs_file):
        raise ValueError(f"Pairs file does not exist: {args.pairs_file}")
    
    print("Starting LFW evaluation...")
    print(f"Model path: {args.model_path}")
    print(f"Data root: {args.data_root}")
    print(f"Pairs file: {args.pairs_file}")
    print(f"Use flip augmentation: {args.use_flip}")
    
    main(args)
