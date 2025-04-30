from abc import ABC, abstractmethod
from typing import Callable

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from .torch_utils import DataLoaders


class Model(ABC):
    """Abstract base class for all models.

    This class defines the interface that all model trainers should implement.
    Specific model trainers should inherit from this class and implement all abstract methods.
    """

    def __init__(self) -> None:
        """Initialize the base model with common attributes."""
        self._criterion = None
        self._optimizer = None

    @property
    @abstractmethod
    def model(self) -> nn.Module:
        """Returns the model to be trained.

        The model parameters should be hardcoded and not overridable by instantiation.

        Returns:
            torch.nn.Module: The model to be trained.
        """
        pass

    def load_model(self, weights_path: str) -> nn.Module:
        """Load model weights from specified path.

        Args:
            weights_path (str): Path to the saved model weights.

        Returns:
            nn.Module: The model with loaded weights.
        """
        model_instance = self.model
        model_instance.load_state_dict(torch.load(weights_path))
        model_instance.eval()
        return model_instance

    @classmethod
    def create_and_load(cls, weights_path: str, **kwargs) -> tuple[nn.Module, "Model"]:
        """Create a model instance, load weights, and return both the model and the model trainer.

        Args:
            weights_path (str): Path to the saved model weights.
            **kwargs: Additional arguments to pass to the model trainer constructor.

        Returns:
            tuple: (model, model_trainer) where model is the loaded nn.Module and
                  model_trainer is the instance of this class.
        """
        model_trainer = cls(**kwargs)
        model = model_trainer.load_model(weights_path)
        return model, model_trainer

    @abstractmethod
    def _train_step(self, data: tuple[torch.Tensor, ...]) -> dict[str, float]:
        """Executes one batch of training.

        Execute the training loop, compute the associated losses, and returns them.

        Args:
            data (tuple[torch.Tensor, ...]): Tuple containing the data for the training step.
            In Causal Inference, the data is usually a tuple (W, A, Y).

        Returns:
            dict[str, float]: Dictionary containing training loss values for the batch.
        """
        pass

    @abstractmethod
    def _val_step(self, data: tuple[torch.Tensor, ...]) -> dict[str, float]:
        """Executes one batch of validation.

        Compute validation losses without updating model parameters.

        Args:
            data (tuple[torch.Tensor, ...]): Tuple containing the data for the validation step.
            In Causal Inference, the data is usually a tuple (W, A, Y).

        Returns:
            dict[str, float]: Dictionary containing validation loss values for the batch.
        """
        pass

    def train_loop(self) -> dict[str, float]:
        """Wrapper executor of one epoch of training and validation.

        This method should perform forward/backward pass, optimize parameters,
        log losses for tracking, and perform validation if a validation dataloader is available.

        Returns:
            dict[str, float]: Dictionary containing loss values for the epoch.
        """
        if not hasattr(self, "dataloaders") or self.dataloaders is None:
            self.dataloaders = self.get_dataloaders()

        # Training phase
        self.model.train()
        train_losses: list[dict[str, float]] = []
        for data in self.dataloaders.train:
            batch_losses = self._train_step(data)
            train_losses.append(batch_losses)

        epoch_train_losses = {}
        for key in train_losses[0].keys():
            epoch_train_losses[key] = sum(batch[key] for batch in train_losses) / len(
                train_losses
            )

        renamed_train_losses = {f"train_{k}": v for k, v in epoch_train_losses.items()}

        # Validation phase
        if hasattr(self.dataloaders, "val") and self.dataloaders.val is not None:
            self.model.eval()
            val_losses: list[dict[str, float]] = []

            with torch.no_grad():
                for data in self.dataloaders.val:
                    batch_losses = self._val_step(data)
                    val_losses.append(batch_losses)

            epoch_val_losses = {}
            for key in val_losses[0].keys():
                epoch_val_losses[key] = sum(batch[key] for batch in val_losses) / len(
                    val_losses
                )

            renamed_val_losses = {f"val_{k}": v for k, v in epoch_val_losses.items()}

            combined_losses = {**renamed_train_losses, **renamed_val_losses}

            self.save_loss(combined_losses)

            return combined_losses

        self.save_loss(renamed_train_losses)

        return renamed_train_losses

    @abstractmethod
    def save_loss(self, losses: dict[str, float]) -> None:
        """Save losses from training loop to training history.

        Args:
            losses (dict[str, float]): Dictionary of losses returned by train_loop.
                          Keys are prefixed with 'train_' or 'val_'.
        """
        pass

    @abstractmethod
    def get_dataloaders(self) -> DataLoaders:
        """Get dataloaders for training and validation.

        Creates and returns dataloaders for both training and validation.

        Returns:
            DataLoaders: A NamedTuple containing:
                - 'train': DataLoader for training
                - 'val': DataLoader for validation
        """
        pass

    @property
    @abstractmethod
    def criterion(self) -> Callable:
        """Returns the loss function for the model.

        Returns:
            callable: The loss function.
        """
        pass

    @property
    @abstractmethod
    def optimizer(self) -> torch.optim.Optimizer:
        """Returns the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        pass

    @abstractmethod
    def get_plots(self) -> list[plt.Figure]:
        """Creates and returns plots for the entire training process.

        Returns:
            list[plt.Figure]: list of matplotlib figures with training plots.
        """
        pass

    @abstractmethod
    def save_model_weights(self, path: str) -> None:
        """Saves the model weights to the specified path.

        Args:
            path (str): Path where model weights should be saved.

        Returns:
            None
        """
        pass
