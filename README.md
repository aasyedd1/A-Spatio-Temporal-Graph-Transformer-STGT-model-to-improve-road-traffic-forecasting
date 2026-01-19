# STGT Model - METR-LA Traffic Forecasting

## Overview
This repository contains the official implementation of the **A Spatio-Temporal Graph Transformer (STGT)
model to improve road traffic forecasting** model for traffic speed forecasting on the METR-LA dataset. The code supports data preprocessing, model training, evaluation, and reproduces results comparable to state-of-the-art methods including STTN/LSTTN.

**Note:** This implementation is based on and extends the following repositories:
- [STTN](https://github.com/xumingxingsjtu/STTN) - Spatial-Temporal Transformer Networks
- [LSTTN](https://github.com/GeoX-Lab/LSTTN) - Long Short-Term Transformer Networks

**Paper:** A Spatio-Temporal Graph Transformer (STGT) model to improve road traffic forecasting  
**Authors:** Sadia Nishat Kazmi, Syed Muhammad Abrar Akber*,Ali Muqtadir  
**Conference/Journal:** Springer nature- Computer Science

## Key Features
- Spatio-temporal graph transformer architecture with uncertainty quantification
- Optimized for NVIDIA RTX GPUs with mixed precision training
- Comprehensive evaluation metrics (MAE, RMSE, MAPE)
- Checkpointing and experiment tracking
- Reproducible results with configurable hyperparameters

## Requirements
- Python >= 3.8
- PyTorch >= 1.12.0
- CUDA-compatible GPU (recommended: 12GB+ VRAM)
- See `requirements.txt` for full dependencies

## Installation

1. **Clone the repository:**
   ```bash
   git clone [your-repo-url]
   cd USTGT_METR-LA
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

### METR-LA Dataset
The METR-LA dataset contains traffic speed data from 207 sensors on highways in Los Angeles County.

**Option 1: Download Original Dataset**
1. Download the METR-LA dataset from the STTN repository:
   - **Primary source:** https://github.com/xumingxingsjtu/STTN (check the data folder or releases)
   - Alternative source: https://github.com/liyaguang/DCRNN
2. Create the following directory structure:
   ```
   ../datasets/
   └── METR-LA/
       ├── data.pkl          # Traffic speed data
       ├── adj_mx.pkl        # Adjacency matrix (optional)
       └── distances.csv     # Sensor distances (optional)
   ```

**Option 2: Generate Dummy Data for Testing**
```bash
cd data_processing
python prepare_datasets.py
```
This will create synthetic data in the correct format for testing the code.

## Usage

### Training
Train the USTGT model with default configuration:
```bash
python train.py --config config.yaml
```

The training script will:
- Load and preprocess the METR-LA dataset
- Train the model with specified hyperparameters
- Save checkpoints to `ustgt_outputs/checkpoints/`
- Log training progress to `ustgt_outputs/checkpoints/training_log.csv`

**Key training parameters** (edit in `config.yaml`):
- `epochs`: Number of training epochs (default: 50)
- `batch_size`: Training batch size (default: 16)
- `learning_rate`: Initial learning rate (default: 0.001)
- `hidden_dim`: Model hidden dimension (default: 64)
- `num_heads`: Number of attention heads (default: 8)

### Evaluation
Evaluate a trained model on the test set:
```bash
python test.py --checkpoint ustgt_outputs/checkpoints/best_model.pth --config config.yaml --output_dir ustgt_outputs/results
```

This will compute:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

Results are saved to `ustgt_outputs/results/`

### Comprehensive Evaluation
For additional analysis including visualizations:
```bash
python eval.py --checkpoint ustgt_outputs/checkpoints/best_model.pth --config config.yaml --output_dir evaluation_results
```

## Project Structure
```
USTGT_METR-LA/
├── train.py                 # Main training script
├── test.py                  # Model evaluation script
├── eval.py                  # Comprehensive evaluation utilities
├── model.py                 # USTGT model architecture
├── config.yaml              # Experiment configuration
├── requirements.txt         # Python dependencies
├── data_processing/         # Data preparation scripts
│   ├── prepare_datasets.py  # Dataset generation
│   └── create_pemsd7m.py    # Additional dataset utilities
├── utils/                   # Utility functions
│   └── metrics.py           # Evaluation metrics
├── utils_common/            # Common utilities
│   ├── __init__.py
│   └── metrics.py
├── ustgt_outputs/           # Training outputs
│   ├── checkpoints/         # Model checkpoints
│   └── results/             # Evaluation results
└── results/                 # Additional results and logs
```

## Expected Results

On METR-LA dataset (12 timesteps prediction):
- MAE: ~[add your results]
- RMSE: ~[add your results]  
- MAPE: ~[add your results]%

Pre-trained model checkpoints are available in `ustgt_outputs/checkpoints/`

## Configuration

Key configuration parameters in `config.yaml`:

**Data:**
- `dataset_name`: METR-LA
- `num_nodes`: 207 (number of sensors)
- `seq_len`: 12 (input sequence length)
- `pred_len`: 12 (prediction horizon)

**Model Architecture:**
- `hidden_dim`: Hidden dimension size
- `num_heads`: Number of attention heads
- `num_layers`: Number of transformer layers
- `dropout`: Dropout rate

**Training:**
- `epochs`: Training epochs
- `batch_size`: Batch size
- `learning_rate`: Learning rate
- `optimizer`: Optimizer type (RMSprop/Adam)

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{your_paper_2024,
  title={[Your Paper Title]},
  author={[Author Names]},
  booktitle={[Conference/Journal Name]},
  year={2024}
}
```

Also consider citing the baseline methods that this work builds upon:
```bibtex
@inproceedings{sttn_2021,
  title={Spatial-Temporal Transformer Networks for Traffic Flow Forecasting},
  author={Xu, Mingxing and Dai, Wenrui and Liu, Chunmiao and Gao, Xing and Lin, Weiyao and Qi, Guo-Jun and Xiong, Hongkai},
  booktitle={arXiv preprint arXiv:2001.02908},
  year={2021}
}

@inproceedings{lsttn_2022,
  title={Long Short-Term Transformer for Online Action Detection},
  author={[LSTTN Authors]},
  booktitle={[Conference]},
  year={2022}
}

@inproceedings{dcrnn_2018,
  title={Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting},
  author={Li, Yaguang and Yu, Rose and Shahabi, Cyrus and Liu, Yan},
  booktitle={ICLR},
  year={2018}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: [your-email@domain.com]

## Acknowledgments

This implementation is based on and extends the code from:
- **STTN** (Spatial-Temporal Transformer Networks): https://github.com/xumingxingsjtu/STTN
- **LSTTN** (Long Short-Term Transformer Networks): https://github.com/GeoX-Lab/LSTTN

We thank the authors of these repositories for making their code publicly available. This work also builds upon the METR-LA dataset and evaluation protocols from DCRNN and related traffic forecasting research.

Special thanks to:
- Mingxing Xu et al. for the STTN implementation and dataset preprocessing scripts
- The GeoX-Lab team for the LSTTN architecture and improvements
- Yaguang Li et al. for the METR-LA dataset (DCRNN)
