import torch
import torch.optim as optim



class VAETrainer:
    def __init__(self, model, optimizer, loss_fn, device, config):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.config = config

    def train(self, dataloader):
        self.model.train()
        for epoch in range(self.config['epochs']):
            total_loss = 0
            for batch in dataloader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                recon, mu, logvar = self.model(batch)
                loss = self.loss_fn(recon, batch, mu, logvar)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path))