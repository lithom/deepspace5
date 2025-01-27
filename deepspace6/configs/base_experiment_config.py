from abc import ABC, abstractmethod

from deepspace6.configs.ds_constants import DeepSpaceConstants
from deepspace6.configs.ds_settings import DeepSpaceSettings
from deepspace6.data.molecule_dataset import MoleculeDatasetHelper


class BaseModelConfig:
    def __init__(self, molecule_dataset_helper: MoleculeDatasetHelper, ds_settings: DeepSpaceSettings, ds_constants = DeepSpaceConstants(MAX_ATOMS=32, DIST_SCALING=1.0, device="cuda") ):
        self.molecule_dataset_helper = molecule_dataset_helper
        self.ds_settings = ds_settings
        self.ds_constants = ds_constants


class GeometryModelConfig:
    def __init__(self, ds_settings: DeepSpaceSettings, ds_constants = DeepSpaceConstants(MAX_ATOMS=32, DIST_SCALING=1.0, device="cuda") ):
        self.molecule_dataset_helper = molecule_dataset_helper
        self.ds_settings = ds_settings
        self.ds_constants = ds_constants


class BaseDataConfig:
    def __init__(self, input_type="smiles", file_train="", file_val=""):
        self.INPUT_TYPE = input_type
        self.FILE_TRAIN = file_train
        self.FILE_VAL   = file_val

class BaseTrainConfig:
    def __init__(self,batch_size=256,learning_rate=0.001,num_epochs=200,grad_clip=1.0,device="cuda"):
        # Training parameters
        self.NUM_EPOCHS = num_epochs
        self.BATCH_SIZE = batch_size
        self.LEARNING_RATE = learning_rate
        #WEIGHT_DECAY = 1e-5
        self.GRAD_CLIP = grad_clip
        self.device = device



class BaseExperimentConfig(ABC):
    # def __init__(self, ds_settings, ds_constants):
    #     """
    #     Initialize the base experiment configuration with shared settings and constants.
    #     """
    #     self.ds_settings = ds_settings
    #     self.ds_constants = ds_constants
    #     self.model_config = None
    #     self.data_config = None
    #     self.train_config = None
    # def __init__(self):


    @abstractmethod
    def create_model_config(self):
        """
        Abstract method to create and return a model configuration.
        """
        pass

    @abstractmethod
    def create_data_config(self):
        """
        Abstract method to create and return a data configuration.
        """
        pass

    @abstractmethod
    def create_train_config(self):
        """
        Abstract method to create and return a training configuration.
        """
        pass

    def setup(self):
        """
        Initialize all configurations. This ensures derived classes properly implement the methods.
        """
        self.model_config = self.create_model_config()
        self.data_config = self.create_data_config()
        self.train_config = self.create_train_config()



