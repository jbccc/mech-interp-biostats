import argparse
import os
import subprocess
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models import ToyTMLENoEffectConfounder, ToyTMLEStrongConfounder


def train(args):
    """Training function reused from the trainer.py file"""
    train_module = None

    match args.model:
        case "tmle_no_effect_confounder":
            train_module = ToyTMLENoEffectConfounder()

        case "tmle_strong_confounder":
            train_module = ToyTMLEStrongConfounder()

        case _:
            raise ValueError(f"Unknown model: {args.model}")

    training_history = []

    for epoch in range(args.epochs):
        epoch_results = train_module.train_loop()
        training_history.append(epoch_results)
        print(
            f"Epoch {epoch + 1}/{args.epochs} - {' | '.join(f'{k}: {v:5f}' for k, v in epoch_results.items())}"
        )
    print(train_module.model)

    if args.plot:
        plots = train_module.get_plots()
        save_dir = f"training_runs/{args.model}"
        os.makedirs(save_dir, exist_ok=True)
        for i, plot in enumerate(plots):
            plot.savefig(f"{save_dir}/plot_{i}.png")
            plt.show()
            plt.close(plot)

    if args.save_model:
        if args.save_path:
            os.makedirs(args.save_path, exist_ok=True)
        train_module.save_model_weights(args.save_path)
        print(f"Model saved to {args.save_path}")


def run_module(module_path):
    """Run a Python module by executing it as a subprocess to capture all output"""
    print(f"Running {module_path}...")

    try:
        subprocess.run(
            [sys.executable, module_path],
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error running {module_path}. Exit code: {e.returncode}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False


def run_experiments(args):
    """Run specified experiment files"""
    experiment_mapping = [
        "experiments/tmle_estimate_interp.py",
        "experiments/pathway_analysis.py",
        "experiments/visualize_causal_pathways.py",
    ]

    experiment_files = []

    if getattr(args, f"all", False):
        experiment_files.extend(experiment_mapping[0:3])

    for i in range(len(experiment_mapping)):
        if (
            getattr(args, f"exp_{i + 1}", False)
            and experiment_mapping[i] not in experiment_files
        ):
            experiment_files.append(experiment_mapping[i])

    if not experiment_files:
        print(
            "No experiments specified to run. Use --all, or --exp-N options."
        )
        return

    # Run each selected experiment
    for exp_file in experiment_files:
        run_module(exp_file)


def main():
    parser = argparse.ArgumentParser(
        description="Train models or run experiments for causal inference research."
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Training command parser
    train_parser = subparsers.add_parser("train", help="Train a specified model")
    train_parser.add_argument(
        "--model", type=str, required=True, help="Name of the model to train"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train"
    )
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    train_parser.add_argument(
        "--plot", action="store_true", help="Plot training losses"
    )
    train_parser.add_argument(
        "--save_model", action="store_true", help="Save model weights after training"
    )
    train_parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Directory to save model weights",
    )

    # Run experiments command parser
    run_parser = subparsers.add_parser("run", help="Run specified experiments")
    run_parser.add_argument(
        "--all",
        action="store_true",
        help="Run all experiments in experiments/ directory",
    )
    run_parser.add_argument(
        "--exp-1",
        action="store_true",
        help="Run TMLE estimate interpretation experiment",
    )
    run_parser.add_argument(
        "--exp-2", action="store_true", help="Run pathway analysis experiment"
    )
    run_parser.add_argument(
        "--exp-3", action="store_true", help="Run visualize causal pathways experiment"
    )

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "run":
        run_experiments(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
