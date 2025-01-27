import os
from pathlib import Path

class DeepSpaceSettings:
    def __init__(self, base_dir):
        # Initialize base directory from the constructor argument
        self.BASE_DIR = Path(base_dir).resolve()
        self.DATA_DIR = self.BASE_DIR / "data"
        self.MODELS_DIR = self.BASE_DIR / "models"
        self.LOGS_DIR = self.BASE_DIR / "logs"
        self.CHECKPOINTS_DIR = self.MODELS_DIR / "checkpoints"

        # Dataset settings
        self.TRAIN_DATA_FILE = self.DATA_DIR / "train_molecules.smi"
        self.VALIDATION_DATA_FILE = self.DATA_DIR / "val_molecules.smi"
        self.TEST_DATA_FILE = self.DATA_DIR / "test_molecules.smi"

        # Model settings
        self.MODEL_NAME = "deepspace_model"
        self.CHECKPOINT_FILE = self.CHECKPOINTS_DIR / f"{self.MODEL_NAME}_best.pt"
        self.LATEST_CHECKPOINT_FILE = self.CHECKPOINTS_DIR / f"{self.MODEL_NAME}_latest.pt"

        # Logging
        self.LOG_FILE = self.LOGS_DIR / f"{self.MODEL_NAME}_training.log"
        self.VERBOSE = True

        # Random seed
        self.RANDOM_SEED = 42

    def create_directories(self):
        """
        Ensure necessary directories exist.
        """
        dirs = [self.DATA_DIR, self.MODELS_DIR, self.LOGS_DIR, self.CHECKPOINTS_DIR]
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
