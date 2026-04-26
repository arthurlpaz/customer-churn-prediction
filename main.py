import argparse
from src.pipelines.train_pipeline import run_training_pipeline

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the training pipeline.")

    parser.add_argument(
        "--config-path", default="src/config/config.yaml", help="Path to the configuration file."
    )

    parser.add_argument(
        "--data-path", default=None, help="Optional data path override (useful for CI/CD)"
    )

    args = parser.parse_args()

    run_training_pipeline(config_path=args.config_path, data_path_override=args.data_path)
