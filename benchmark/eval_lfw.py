#!/usr/bin/env python3
"""
Simplified LFW Benchmark Script for LVFace ONNX Model
Uses the LVFaceONNXInferencer class for clean and efficient evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
import argparse
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from tqdm import tqdm
import warnings
import csv
import prettytable
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference_onnx import LVFaceONNXInferencer


class LVFaceONNXLFWBenchmark:
    """LFW benchmark evaluation for LVFace ONNX model using the inference class."""
    
    def __init__(self, model_path: str, lfw_data_path: str, use_gpu: bool = True):
        """
        Initialize LFW benchmark.
        
        Args:
            model_path: Path to LVFace ONNX model file
            lfw_data_path: Path to LFW dataset directory
            use_gpu: Whether to use GPU acceleration
        """
        self.lfw_data_path = Path(lfw_data_path)
        self.image_dir = self.lfw_data_path / 'lfw-deepfunneled' / 'lfw-deepfunneled'
        
        # Initialize LVFace ONNX inferencer
        print(f"Loading LVFace ONNX model from {model_path}")
        self.inferencer = LVFaceONNXInferencer(model_path, use_gpu=use_gpu)
        
        print(f"LFW image directory: {self.image_dir}")
        print(f"Using GPU: {use_gpu}")
        
    def get_image_path(self, person_name: str, image_num: int) -> Path:
        """
        Get path to LFW image.
        
        Args:
            person_name: Name of the person
            image_num: Image number for the person
            
        Returns:
            Path to the image file
        """
        image_filename = f"{person_name}_{image_num:04d}.jpg"
        return self.image_dir / person_name / image_filename
    
    def load_lfw_pairs(self) -> tuple:
        """
        Load LFW pairs from CSV file.
        
        Returns:
            Tuple of (pairs_data, labels) where:
            - pairs_data: List of tuples (person1, img1, person2, img2)
            - labels: List of labels (1 for same person, 0 for different)
        """
        pairs_file = self.lfw_data_path / 'pairs.csv'
        
        if not pairs_file.exists():
            # Try alternative locations
            pairs_file = self.lfw_data_path / 'pairs.txt'
            if not pairs_file.exists():
                raise FileNotFoundError(f"Pairs file not found. Expected at: {self.lfw_data_path / 'pairs.csv'}")
        
        pairs_data = []
        labels = []
        
        print(f"Loading LFW pairs from {pairs_file}...")
        
        with open(pairs_file, 'r') as f:
            lines = f.readlines()
            
        # Skip header if it exists
        start_idx = 1 if 'person' in lines[0].lower() or 'name' in lines[0].lower() else 0
            
        for line in tqdm(lines[start_idx:], desc="Processing pairs"):
            line = line.strip()
            if not line:
                continue
            
            # Handle both comma and tab/space separated formats
            if ',' in line:
                parts = [part.strip() for part in line.split(',') if part.strip()]
            else:
                parts = [part.strip() for part in line.split() if part.strip()]
            
            if len(parts) == 3:
                # Positive pair: same person, different images
                # Format: name, img1_num, img2_num
                person1 = parts[0]
                img1_num = int(parts[1])
                person2 = parts[0]  # Same person
                img2_num = int(parts[2])
                
                pairs_data.append((person1, img1_num, person2, img2_num))
                labels.append(1)  # Same person
                
            elif len(parts) == 4:
                # Negative pair: different people
                # Format: person1_name, img1_num, person2_name, img2_num
                person1 = parts[0]
                img1_num = int(parts[1])
                person2 = parts[2]
                img2_num = int(parts[3])
                
                pairs_data.append((person1, img1_num, person2, img2_num))
                labels.append(0)  # Different people
            else:
                print(f"Skipping malformed line: {line}")
                continue
        
        print(f"Loaded {len(pairs_data)} pairs ({sum(labels)} positive, {len(labels) - sum(labels)} negative)")
        return pairs_data, labels
    
    def evaluate_pairs(self, pairs_data: list, labels: list) -> tuple:
        """
        Evaluate model on LFW pairs.
        
        Args:
            pairs_data: List of image pairs
            labels: Ground truth labels
            
        Returns:
            Tuple of (similarities, labels, skipped_pairs)
        """
        similarities = []
        valid_labels = []
        skipped_pairs = 0
        
        print("Computing embeddings and similarities...")
        for i, (person1, img1, person2, img2) in enumerate(tqdm(pairs_data)):
            try:
                # Get image paths
                image1_path = self.get_image_path(person1, img1)
                image2_path = self.get_image_path(person2, img2)
                
                # Check if images exist
                if not image1_path.exists():
                    raise FileNotFoundError(f"Image 1 not found: {image1_path}")
                if not image2_path.exists():
                    raise FileNotFoundError(f"Image 2 not found: {image2_path}")
                
                # Get embeddings using the inferencer
                emb1 = self.inferencer.infer_from_image(str(image1_path))
                emb2 = self.inferencer.infer_from_image(str(image2_path))
                
                # Compute similarity using the inferencer's method
                similarity = self.inferencer.calculate_similarity(emb1, emb2)
                similarities.append(similarity)
                valid_labels.append(labels[i])
                
            except (FileNotFoundError, Exception) as e:
                print(f"Skipping pair {i+1}: {e}")
                skipped_pairs += 1
                continue
        
        print(f"Processed {len(similarities)} pairs, skipped {skipped_pairs} pairs")
        return np.array(similarities), np.array(valid_labels), skipped_pairs
    
    def compute_metrics(self, similarities: np.ndarray, labels: np.ndarray) -> dict:
        """
        Compute evaluation metrics.
        
        Args:
            similarities: Array of similarity scores
            labels: Array of ground truth labels
            
        Returns:
            Dictionary of metrics
        """
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(labels, similarities)
        roc_auc = auc(fpr, tpr)
        
        # Find optimal threshold (Youden's J statistic)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        # Compute accuracy at optimal threshold
        predictions = (similarities >= optimal_threshold).astype(int)
        accuracy = accuracy_score(labels, predictions)
        
        # Compute true positive rate and false positive rate at optimal threshold
        optimal_tpr = tpr[optimal_idx]
        optimal_fpr = fpr[optimal_idx]
        
        # Additional metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)
        
        # Compute TAR at specific FAR values (IJBC-style metrics)
        tar_at_far = self.compute_tar_at_far(fpr, tpr)
        
        metrics = {
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'optimal_threshold': optimal_threshold,
            'optimal_tpr': optimal_tpr,
            'optimal_fpr': optimal_fpr,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'tar_at_far': tar_at_far
        }
        
        return metrics
    
    def compute_tar_at_far(self, fpr: np.ndarray, tpr: np.ndarray, 
                          far_values: list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]) -> dict:
        """
        Compute True Accept Rate (TAR) at specific False Accept Rate (FAR) values.
        This follows the same methodology as IJBC benchmark.
        
        Args:
            fpr: False Positive Rate array from ROC curve
            tpr: True Positive Rate array from ROC curve
            far_values: List of FAR values to evaluate at
            
        Returns:
            Dictionary mapping FAR values to TAR values
        """
        tar_at_far = {}
        
        # Flip arrays to match IJBC implementation (descending order)
        fpr_flipped = np.flipud(fpr)
        tpr_flipped = np.flipud(tpr)
        
        for far in far_values:
            # Find the closest FPR to the target FAR
            abs_diff = np.abs(fpr_flipped - far)
            min_index = np.argmin(abs_diff)
            tar_at_far[far] = tpr_flipped[min_index]
            
        return tar_at_far
    
    def create_tar_far_table(self, tar_at_far: dict) -> prettytable.PrettyTable:
        """
        Create a formatted table showing TAR at different FAR values.
        
        Args:
            tar_at_far: Dictionary mapping FAR values to TAR values
            
        Returns:
            PrettyTable with TAR@FAR results
        """
        far_labels = [f'1e-{i}' for i in range(6, 0, -1)]  # [1e-6, 1e-5, ..., 1e-1]
        table = prettytable.PrettyTable(['Method'] + far_labels)
        
        row = ['LVFace-LFW']
        for far in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            tar = tar_at_far.get(far, 0.0)
            row.append(f'{tar * 100:.2f}')  # Convert to percentage
            
        table.add_row(row)
        return table
    
    def plot_results(self, metrics: dict, similarities: np.ndarray, labels: np.ndarray, save_path: str = None):
        """
        Plot comprehensive evaluation results.
        
        Args:
            metrics: Computed metrics dictionary
            similarities: Array of similarity scores
            labels: Array of ground truth labels
            save_path: Path to save plots (optional)
        """
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(24, 12))
        
        # ROC Curve
        ax1.plot(metrics['fpr'], metrics['tpr'], 'b-', linewidth=2,
                label=f'LVFace (AUC = {metrics["roc_auc"]:.4f})')
        ax1.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
        ax1.plot(metrics['optimal_fpr'], metrics['optimal_tpr'], 'ro', markersize=8,
                label=f'Optimal Point (t = {metrics["optimal_threshold"]:.3f})')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve - LVFace on LFW Dataset')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Similarity Distribution
        pos_similarities = similarities[labels == 1]
        neg_similarities = similarities[labels == 0]
        
        ax2.hist(neg_similarities, bins=50, alpha=0.7, label=f'Different People (n={len(neg_similarities)})', 
                color='red', density=True)
        ax2.hist(pos_similarities, bins=50, alpha=0.7, label=f'Same Person (n={len(pos_similarities)})', 
                color='blue', density=True)
        ax2.axvline(metrics['optimal_threshold'], color='green', linestyle='--', linewidth=2,
                   label=f'Optimal Threshold ({metrics["optimal_threshold"]:.3f})')
        ax2.set_xlabel('Cosine Similarity')
        ax2.set_ylabel('Density')
        ax2.set_title('Similarity Score Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # TAR@FAR Plot
        far_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        tar_values = [metrics['tar_at_far'][far] * 100 for far in far_values]
        far_labels = [f'1e-{i}' for i in range(6, 0, -1)]
        
        bars = ax3.bar(far_labels, tar_values, color='skyblue', alpha=0.7, edgecolor='navy')
        ax3.set_xlabel('False Accept Rate (FAR)')
        ax3.set_ylabel('True Accept Rate (TAR) %')
        ax3.set_title('TAR at Different FAR Values')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
        
        # Add value labels on top of bars
        for bar, value in zip(bars, tar_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Special highlight for FAR=1e-4
        far_1e4_idx = far_values.index(1e-4)
        bars[far_1e4_idx].set_color('orange')
        bars[far_1e4_idx].set_edgecolor('darkorange')
        bars[far_1e4_idx].set_linewidth(2)
        
        # Accuracy vs Threshold
        accuracies = []
        threshold_range = np.linspace(similarities.min(), similarities.max(), 100)
        for thresh in threshold_range:
            preds = (similarities >= thresh).astype(int)
            acc = accuracy_score(labels, preds)
            accuracies.append(acc)
        
        ax4.plot(threshold_range, accuracies, 'g-', linewidth=2)
        ax4.axvline(metrics['optimal_threshold'], color='red', linestyle='--', linewidth=2,
                   label=f'Optimal Threshold')
        ax4.axhline(metrics['accuracy'], color='red', linestyle='--', linewidth=2,
                   label=f'Max Accuracy ({metrics["accuracy"]:.4f})')
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Accuracy vs Threshold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Confusion Matrix
        predictions = (similarities >= metrics['optimal_threshold']).astype(int)
        cm = confusion_matrix(labels, predictions)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
                   xticklabels=['Different', 'Same'], yticklabels=['Different', 'Same'])
        ax5.set_xlabel('Predicted')
        ax5.set_ylabel('Actual')
        ax5.set_title('Confusion Matrix')
        
        # ROC Curve (Log Scale for FPR)
        ax6.semilogx(metrics['fpr'], metrics['tpr'], 'b-', linewidth=2,
                    label=f'LVFace (AUC = {metrics["roc_auc"]:.4f})')
        ax6.semilogx([1e-6, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
        
        # Highlight specific FAR points
        for far in far_values:
            tar = metrics['tar_at_far'][far]
            if far >= metrics['fpr'].min():  # Only plot if within range
                ax6.plot(far, tar, 'ro', markersize=6)
                ax6.annotate(f'FAR={far:.0e}\nTAR={tar:.3f}', 
                           (far, tar), xytext=(5, 5), textcoords='offset points',
                           fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        ax6.set_xlabel('False Accept Rate (FAR) - Log Scale')
        ax6.set_ylabel('True Accept Rate (TAR)')
        ax6.set_title('ROC Curve - Log Scale FPR')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_xlim(1e-6, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Results saved to {save_path}")
        
        plt.show()
    
    def run_benchmark(self, save_results: bool = True) -> dict:
        """
        Run complete LFW benchmark evaluation.
        
        Args:
            save_results: Whether to save results to files
            
        Returns:
            Dictionary of evaluation results
        """
        print("=" * 60)
        print("LVFace ONNX LFW Benchmark Evaluation")
        print("=" * 60)
        
        # Load LFW pairs
        pairs_data, labels = self.load_lfw_pairs()
        
        # Evaluate pairs
        similarities, valid_labels, skipped_pairs = self.evaluate_pairs(pairs_data, labels)
        
        if len(similarities) == 0:
            raise ValueError("No valid pairs found for evaluation!")
        
        # Compute metrics
        metrics = self.compute_metrics(similarities, valid_labels)
        
        # Print results
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Model Type: LVFace ONNX")
        print(f"Total pairs processed: {len(similarities)}")
        print(f"Pairs skipped: {skipped_pairs}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"Best Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"Optimal Threshold: {metrics['optimal_threshold']:.4f}")
        print(f"True Positive Rate: {metrics['optimal_tpr']:.4f}")
        print(f"False Positive Rate: {metrics['optimal_fpr']:.4f}")
        
        # Print TAR@FAR table (IJBC-style metrics)
        print("\n" + "=" * 60)
        print("TAR @ FAR EVALUATION (IJBC-style)")
        print("=" * 60)
        tar_far_table = self.create_tar_far_table(metrics['tar_at_far'])
        print(tar_far_table)
        
        # Highlight specific FAR values
        print(f"\nKey Metrics:")
        print(f"TAR @ FAR=1e-4: {metrics['tar_at_far'][1e-4]*100:.2f}%")
        print(f"TAR @ FAR=1e-3: {metrics['tar_at_far'][1e-3]*100:.2f}%")
        print(f"TAR @ FAR=1e-2: {metrics['tar_at_far'][1e-2]*100:.2f}%")
        
        # Plot results
        if save_results:
            save_dir = Path('results')
            save_dir.mkdir(exist_ok=True)
            save_path = save_dir / 'lfw_onnx_evaluation_results.png'
        else:
            save_path = None
            
        self.plot_results(metrics, similarities, valid_labels, save_path)
        
        # Save detailed results
        if save_results:
            results_file = save_dir / 'lfw_onnx_benchmark_results.npz'
            np.savez(results_file,
                    similarities=similarities,
                    labels=valid_labels,
                    **{k: v for k, v in metrics.items() if isinstance(v, (int, float, np.ndarray))})
            print(f"Detailed results saved to {results_file}")
            
            # Save TAR@FAR table to CSV (IJBC-style)
            csv_file = save_dir / 'lfw_onnx_tar_far_results.csv'
            tar_far_table = self.create_tar_far_table(metrics['tar_at_far'])
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow(['Method', '1e-6', '1e-5', '1e-4', '1e-3', '1e-2', '1e-1'])
                # Write data row
                row = ['LVFace-LFW']
                for far in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
                    tar = metrics['tar_at_far'].get(far, 0.0)
                    row.append(f'{tar * 100:.2f}')
                writer.writerow(row)
            print(f"TAR@FAR results saved to {csv_file}")
            
            # Save summary to text file
            summary_file = save_dir / 'lfw_onnx_summary.txt'
            with open(summary_file, 'w') as f:
                f.write("LVFace ONNX LFW Benchmark Results\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Total pairs processed: {len(similarities)}\n")
                f.write(f"Pairs skipped: {skipped_pairs}\n")
                f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n")
                f.write(f"Best Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"Precision: {metrics['precision']:.4f}\n")
                f.write(f"Recall: {metrics['recall']:.4f}\n")
                f.write(f"F1-Score: {metrics['f1_score']:.4f}\n")
                f.write(f"Optimal Threshold: {metrics['optimal_threshold']:.4f}\n")
                f.write(f"True Positive Rate: {metrics['optimal_tpr']:.4f}\n")
                f.write(f"False Positive Rate: {metrics['optimal_fpr']:.4f}\n")
                f.write("\nTAR @ FAR Results:\n")
                f.write("=" * 30 + "\n")
                for far in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
                    tar = metrics['tar_at_far'][far]
                    f.write(f"TAR @ FAR={far:.0e}: {tar*100:.2f}%\n")
            print(f"Summary saved to {summary_file}")
        
        return {
            'similarities': similarities,
            'labels': valid_labels,
            'metrics': metrics,
            'skipped_pairs': skipped_pairs
        }


def main():
    """Main function for running LFW benchmark."""
    parser = argparse.ArgumentParser(description='LFW Benchmark for LVFace ONNX Model')
    parser.add_argument('--model_path', type=str, 
                       default='models/LVFace-L_Glint360K.onnx',
                       help='Path to LVFace ONNX model file')
    parser.add_argument('--lfw_data_path', type=str,
                       default='data/lfw',
                       help='Path to LFW dataset directory')
    parser.add_argument('--no_gpu', action='store_true',
                       help='Disable GPU acceleration')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save results to files')
    
    args = parser.parse_args()
    
    # Validate paths
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model file not found: {model_path}")
    
    lfw_path = Path(args.lfw_data_path)
    if not lfw_path.exists():
        raise FileNotFoundError(f"LFW dataset not found: {lfw_path}")
    
    # Run benchmark
    benchmark = LVFaceONNXLFWBenchmark(
        model_path=str(model_path),
        lfw_data_path=str(lfw_path),
        use_gpu=not args.no_gpu
    )
    
    results = benchmark.run_benchmark(save_results=not args.no_save)
    
    print("\nBenchmark completed successfully!")
    print(f"Final ROC AUC: {results['metrics']['roc_auc']:.4f}")
    print(f"Final Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"TAR @ FAR=1e-4: {results['metrics']['tar_at_far'][1e-4]*100:.2f}%")
    
    return results


if __name__ == '__main__':
    main()