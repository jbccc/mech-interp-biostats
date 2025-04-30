import os
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data.no_effect_confounders_data import \
    generate_no_effect_confounders_data
from src.utils.abstract_model import Model
from src.utils.torch_utils import DataLoaders
from src.utils.toy_tmle_model import ToyTMLE


class ToyTMLENoEffectConfounder(Model):
    """Trainer for the Toy TMLE model with No Effect Confounders."""

    def __init__(self) -> None:
        """Initialize the ToyTMLENoEffectConfounder model."""
        super().__init__()
        self.seed = 44
        self.n_samples = 10000
        self.test_size = 0.2
        self.batch_size = 128

        self.dataloaders = None
        self.scaler = None

        self._model = None
        self._optimizer = None

        self.input_dim = 6
        self.hidden_layers = 5
        self.hidden_size = 50
        self.alpha = 0.5
        self.learning_rate = 0.0003
        self.weight_decay = 1e-5

        self.model_dir = "./checkpoints/toy_tmle/no_effect_confounder"
        self.model_weights_path = f"{self.model_dir}/best.pth"

        self.training_history = {
            "train_losses": [],
            "val_losses": [],
            "train_losses_q": [],
            "val_losses_q": [],
            "train_losses_g": [],
            "val_losses_g": [],
        }

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    @property
    def model(self) -> ToyTMLE:
        """Returns the TMLE No Effect Confounder model with hardcoded parameters.

        Returns:
            torch.nn.Module: The TMLE No Effect Confounder model.
        """
        if self._model is None:
            self._model = ToyTMLE(
                input_dim=self.input_dim,
                hidden_layers=self.hidden_layers,
                hidden_size=self.hidden_size,
            )
        return self._model

    def get_dataloaders(self) -> DataLoaders:
        """Get dataloaders for training and validation.

        Returns:
            DataLoaders: A NamedTuple containing train and validation dataloaders
        """
        if self.dataloaders is None:
            print("Loading and preparing data (No Effect Confounders)...")
            data, feature_cols = generate_no_effect_confounders_data(
                n_samples=self.n_samples, seed=self.seed
            )

            W = data[feature_cols].values
            A = data["A"].values
            Y = data["Y"].values

            W_train, W_val, A_train, A_val, Y_train, Y_val = train_test_split(
                W, A, Y, test_size=self.test_size, random_state=self.seed, stratify=A
            )

            self.scaler = StandardScaler()
            W_train_scaled = self.scaler.fit_transform(W_train)
            W_val_scaled = self.scaler.transform(W_val)

            train_dataset = TensorDataset(
                torch.FloatTensor(W_train_scaled),
                torch.FloatTensor(A_train).unsqueeze(1),
                torch.FloatTensor(Y_train).unsqueeze(1),
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(W_val_scaled),
                torch.FloatTensor(A_val).unsqueeze(1),
                torch.FloatTensor(Y_val).unsqueeze(1),
            )

            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )

            self.dataloaders = DataLoaders(train=train_loader, val=val_loader)

            print(
                f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
            )

        return self.dataloaders

    def _train_step(
        self, data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, float]:
        """Execute one batch of training.

        Args:
            data (tuple): Tuple containing (W, A, Y) tensors.

        Returns:
            dict: Dictionary containing training loss values for the batch.
        """
        w_batch, a_batch, y_batch = data

        self.optimizer.zero_grad()
        y_hat, a_hat = self.model(w_batch, a_batch)
        loss, loss_q, loss_g = self.criterion(
            y_hat, a_hat, y_batch.squeeze(-1), a_batch.squeeze(-1)
        )
        loss.backward()
        self.optimizer.step()

        return {
            "total_loss": loss.item(),
            "q_loss": loss_q.item(),
            "g_loss": loss_g.item(),
        }

    def _val_step(
        self, data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, float]:
        """Execute one batch of validation.

        Args:
            data (tuple): Tuple containing (W, A, Y) tensors.

        Returns:
            dict: Dictionary containing validation loss values for the batch.
        """
        w_batch, a_batch, y_batch = data

        y_hat, a_hat = self.model(w_batch, a_batch)
        loss, loss_q, loss_g = self.criterion(
            y_hat, a_hat, y_batch.squeeze(-1), a_batch.squeeze(-1)
        )

        return {
            "total_loss": loss.item(),
            "q_loss": loss_q.item(),
            "g_loss": loss_g.item(),
        }

    def save_loss(self, losses: dict[str, float]) -> None:
        """Save losses from training loop to training history.

        Args:
            losses (dict): Dictionary of losses returned by train_loop
        """
        # Store training losses
        if "train_total_loss" in losses:
            self.training_history["train_losses"].append(losses["train_total_loss"])
        if "train_q_loss" in losses:
            self.training_history["train_losses_q"].append(losses["train_q_loss"])
        if "train_g_loss" in losses:
            self.training_history["train_losses_g"].append(losses["train_g_loss"])

        # Store validation losses if available
        if "val_total_loss" in losses:
            self.training_history["val_losses"].append(losses["val_total_loss"])
        if "val_q_loss" in losses:
            self.training_history["val_losses_q"].append(losses["val_q_loss"])
        if "val_g_loss" in losses:
            self.training_history["val_losses_g"].append(losses["val_g_loss"])

    @property
    def criterion(
        self,
    ) -> Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Get the criterion function.

        Returns:
            callable: The loss function that takes (y_hat, a_hat, y, a) and returns
                     (total_loss, outcome_loss, propensity_loss).
        """

        def combined_loss(
            y_hat: torch.Tensor, a_hat: torch.Tensor, y: torch.Tensor, a: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Combined loss function for TMLE model.

            Args:
                y_hat (torch.Tensor): Predicted outcomes
                a_hat (torch.Tensor): Predicted propensity scores
                y (torch.Tensor): True outcomes
                a (torch.Tensor): True treatment assignments

            Returns:
                tuple: Contains:
                    - total_loss (torch.Tensor): Combined loss
                    - q_loss (torch.Tensor): Outcome prediction loss
                    - g_loss (torch.Tensor): Propensity score prediction loss
            """
            # Outcome prediction loss (MSE)
            q_loss = F.mse_loss(y_hat, y)

            # Propensity score prediction loss (BCE)
            g_loss = F.binary_cross_entropy(a_hat, a)

            # Combined loss with alpha weighting
            total_loss = (1 - self.alpha) * q_loss + self.alpha * g_loss

            return total_loss, q_loss, g_loss

        return combined_loss

    @property
    def optimizer(self) -> optim.Optimizer:
        """Get the optimizer.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        if self._optimizer is None:
            self._optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        return self._optimizer

    def get_plots(self) -> list[Figure]:
        """Create and return plots for the training process.

        Returns:
            list: List of matplotlib figures with training plots.
        """
        figures = []

        # Check if training history exists and has data
        if not self.training_history or len(self.training_history["train_losses"]) == 0:
            print("No training history found. Run training first.")
            return figures

        # Create overall loss plot
        fig = plt.figure(figsize=(12, 6))
        epochs = range(1, len(self.training_history["train_losses"]) + 1)
        plt.plot(
            epochs, self.training_history["train_losses"], label="Train Total Loss"
        )
        plt.plot(
            epochs,
            self.training_history["val_losses"],
            label="Val Total Loss",
            linestyle="--",
        )
        plt.plot(
            epochs,
            self.training_history["train_losses_q"],
            label="Train Q Loss",
            alpha=0.7,
        )
        plt.plot(
            epochs,
            self.training_history["val_losses_q"],
            label="Val Q Loss",
            linestyle="--",
            alpha=0.7,
        )
        plt.plot(
            epochs,
            self.training_history["train_losses_g"],
            label="Train g Loss",
            alpha=0.7,
        )
        plt.plot(
            epochs,
            self.training_history["val_losses_g"],
            label="Val g Loss",
            linestyle="--",
            alpha=0.7,
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Losses (No Effect Confounders)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        figures.append(fig)

        return figures

    def save_model_weights(self, path: str | None = None) -> None:
        """Save the model weights to the specified path or default path.

        Args:
            path (str, optional): Path where model weights should be saved.
                                 If None, uses the default path. Defaults to None.
        """
        if path is None:
            # Check if there's a custom checkpoint in the directory
            if os.path.exists(self.model_dir):
                # Check for any .pth file that's not best.pth
                for file in os.listdir(self.model_dir):
                    if file.endswith(".pth") and file != "best.pth":
                        path = os.path.join(self.model_dir, file)
                        break

            # If no custom file found, use default
            if path is None:
                path = self.model_weights_path

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model weights saved to {path}")

    def load_model(self, weights_path: str | None = None) -> torch.nn.Module:
        """Load model weights from specified path or find default weights.

        Overrides the abstract Model.load_model method to handle default path logic.

        Args:
            weights_path (str, optional): Path to the saved model weights.
                                         If None, will attempt to find default weights.

        Returns:
            nn.Module: The model with loaded weights.
        """
        model_instance = self.model

        if weights_path is None:
            # Check if there's a custom checkpoint in the directory
            if os.path.exists(self.model_dir):
                # Check for any .pth file that's not best.pth
                for file in os.listdir(self.model_dir):
                    if file.endswith(".pth") and file != "best.pth":
                        weights_path = os.path.join(self.model_dir, file)
                        break

            # If no custom file found, use default
            if weights_path is None:
                weights_path = self.model_weights_path

        if os.path.exists(weights_path):
            model_instance.load_state_dict(torch.load(weights_path))
            print(f"Model weights loaded from {weights_path}")
        else:
            print(
                f"No weights found at {weights_path}. Using randomly initialized weights."
            )

        model_instance.eval()
        return model_instance
