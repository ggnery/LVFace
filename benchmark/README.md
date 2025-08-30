# LVFace Benchmark Scripts

This directory contains evaluation scripts for face recognition benchmarks.

## LFW Evaluation

The `eval_lfw.py` script evaluates face recognition models on the LFW (Labeled Faces in the Wild) dataset.

### Usage

Basic usage:
```bash
python eval_lfw.py --model-path /path/to/onnx/model
```

Full options:
```bash
python eval_lfw.py \
    --model-path ./models/LVFace-L_Glint360K.onnx \
    --data-root ./data/lfw-deepfunneled/lfw-deepfunneled \
    --pairs-file ./data/pairs.csv \
    --batch-size 32 \
    --use-flip \
    --save-results ./results/lfw_results.npy
```

### Arguments

- `--model-path`: Path to ONNX model directory (required)
- `--data-root`: Path to LFW dataset directory (default: `./data/lfw-deepfunneled/lfw-deepfunneled`)
- `--pairs-file`: Path to pairs CSV file (default: `./data/pairs.csv`)
- `--batch-size`: Batch size for feature extraction (default: 32)
- `--num-workers`: Number of data loading workers (default: 4)
- `--use-flip` / `--no-flip`: Enable/disable flip test augmentation (default: enabled)
- `--save-results`: Path to save detailed results (optional)


### Expected Output

The script will output:
- Dataset loading statistics
- Feature extraction progress
- Performance metrics including:
  - AUC (Area Under Curve)
  - EER (Equal Error Rate)
  - Accuracy at best threshold
  - TAR (True Accept Rate) at various FAR (False Accept Rate) levels

### Data Format

The script expects:
- LFW images in directory structure: `{person_name}/{person_name}_{image_number:04d}.jpg`
- CSV files with verification pairs:
  - `pairs.csv`: Same person pairs (format: `name,imagenum1,imagenum2`)
  - `mismatchpairsDevTest.csv`: Different person pairs (format: `name1,imagenum1,name2,imagenum2`)

### Requirements

- PyTorch
- OpenCV
- scikit-learn
- pandas
- numpy
- prettytable
- onnxruntime

Install requirements:
```bash
pip install torch opencv-python scikit-learn pandas numpy prettytable onnxruntime
```

For GPU acceleration, install onnxruntime-gpu instead of onnxruntime.
