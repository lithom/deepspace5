import torch



class OutputHeadConfiguration:
    def __init__(self):
        pass

    def load_config(self, config):
        """
        loads the configuration for this output head
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    def create_module(self):
        """
        Create and return the module (PyTorch layers) for this output head.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def set_global_data(self, head_name, device, global_data):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_data_length(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def has_input_sample(self):
        """
        Return True if this head provides input samples, False otherwise.
        """
        return False

    def get_data_sample(self, idx):
        """
        Create data sample idx
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def compute_loss(self, output, target):
        """
        Given the output from the module and the target data, compute and return the loss.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


    def start_iteration(self):
        """
        Info call. Can be used e.g. to update internal counters, for collecting more detailed performance
        metrics that can be shown at the end of an iteration.
        """

    def end_iteration(self, output_level: int):
        """
        Info call. Can be used e.g. to show additional internal performance metrics at the end of an iteration.
        output_level: if zero, then there should be no output, values > zero should create output (more detailed with
        increasing value)
        """

class ExampleHead(OutputHeadConfiguration):
    def __init__(self):
        super().__init__()
        self.head_name = None
        self.global_data = None

    def load_config(self, config):
        print('load config')

    def create_module(self):
        # Example: Create a simple linear layer as the module
        module = torch.nn.Linear(256, 10)  # Example dimensions
        return module

    def set_global_data(self, head_name, global_data):
        self.head_name = head_name
        self.global_data = global_data

    def get_data_sample(self, idx):
        return torch.randn((24))

    def get_data_length(self):
        return 512

    def has_input_sample(self):
        """
        Return True if this head provides input samples, False otherwise.
        """
        return False


    def compute_loss(self, output, target):
        # Example: Use CrossEntropyLoss for classification
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, target)
        return loss


