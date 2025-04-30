# Causal Inference Master Thesis

This repository contains the code for my master thesis project focused on leveraging Mechanistic Interpretability as a way to improve Causal Inference for Biostats.

## Repository Structure

### Source Code (`src/`)

The `src/` directory is structured as follows:

- **data/**: Contains data loaders and generators for synthetic datasets
- **models/**: Contains deep learning model implementations for causal inference
- **utils/**: Contains utility functions and abstract classes

### Experiments

- **experiments/**: Contains experiment scripts for analysis
  - `tmle_estimate_interp.py`: TMLE estimation and interpretation
  - `pathway_analysis.py`: Analysis of causal pathways
  - `visualize_causal_pathways.py`: Visualization of causal pathways

## Using the Code

### Training Models

To train a model, use the `main.py` script with the `train` command:

```bash
python main.py train --model MODEL_NAME [options]
```

Available models:
- `tmle_no_effect_confounder`: Toy TMLE model with no effect confounders
- `tmle_strong_confounder`: Toy TMLE model with strong confounders

Training options:
- `--epochs`: Number of epochs to train (default: 10)
- `--lr`: Learning rate (default: 0.001)
- `--plot`: Generate and save training plots
- `--save_model`: Save model weights after training 
- `--save_path`: Directory to save model weights (default: *checkpoint/{model_type}/ {experiment}/best.pth*)

Example:
```bash
python main.py train --model tmle_strong_confounder --epochs 20 --plot --save_model --save_path ./checkpoints/toy_tmle/strong_confounder/20_epoch.pth
```

### Running Experiments

To run experiments, use the `main.py` script with the `run` command:

```bash
python main.py run [options]
```

Run options:
- `--all`: Run all main experiments
- `--exp-1`: Run TMLE estimate interpretation experiment
- `--exp-2`: Run pathway analysis experiment
- `--exp-3`: Run visualize causal pathways experiment

Example:
```bash
python main.py run --all  # Run all main experiments
python main.py run --exp-1 --exp-3  # Run experiments 1 and 3
```

## Dependencies

The code uses PyTorch for model implementation and matplotlib for visualization. All dependencies should be installed prior to running the code.

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```
