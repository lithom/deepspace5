base_model_registry = {}

def register_base_model(name):
    def register(cls):
        base_model_registry[name] = cls
        return cls
    return register

class BaseModelConfiguration:
    def __init__(self):
        pass

    def load_base_model_from_config(self, config):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def create_model(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def set_global_data(self, global_data):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def create_input_data_sample(self, idx):
        raise NotImplementedError("This method should be implemented by subclasses.")