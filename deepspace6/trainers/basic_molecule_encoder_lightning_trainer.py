
import torch
import pytorch_lightning as pl

from torch.optim.lr_scheduler import CyclicLR



class LinearRampUpScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, ramp_up_steps, base_lr, last_epoch=-1):
        self.ramp_up_steps = ramp_up_steps
        self.base_lr = base_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.ramp_up_steps:
            return [self.base_lr for _ in self.optimizer.param_groups]
        scale = (self.last_epoch + 1) / self.ramp_up_steps
        return [scale * self.base_lr for _ in self.optimizer.param_groups]


# LightningModule wrapping your model and training logic
class MoleculeLightningModule(pl.LightningModule):
    def __init__(self, model, molecule_dataset_helper, train_config, ds_settings, ramp_up_steps=100):
        super().__init__()
        self.model = model
        self.molecule_dataset_helper = molecule_dataset_helper
        self.train_config = train_config
        self.ds_settings = ds_settings
        self.ramp_up_steps = ramp_up_steps
        self.cosine_total_steps = 1000
        self.cosine_base_lr = train_config.LEARNING_RATE

    def forward(self, atom_data, bond_data):
        # Forward pass simply delegates to the underlying model.
        return self.model(atom_data, bond_data)

    def loss_fn(self, atom_data, bond_data, atom_output, bond_output, batch, is_val, global_idx):
        device = self.train_config.device
        dataset_helper = self.molecule_dataset_helper
        atom_parts = dataset_helper.atom_embedding_parts
        bond_parts = dataset_helper.bond_embedding_parts

        atom_loss = torch.zeros(atom_data.size(0), atom_data.size(1), device=device)
        bond_loss = torch.zeros(bond_data.size(0), bond_data.size(1), device=device)

        val_or_train = "val" if is_val else "train"

        for zi, part in enumerate(atom_parts):
            atom_mask = batch["atom_mask"][zi].to(device)
            part_name = part.__class__.__name__
            atom_target = dataset_helper.prepare_for_loss(atom_data, batch["atom_metadata"], part_name)
            atom_output_part = dataset_helper.prepare_for_loss(atom_output, batch["atom_metadata"], part_name)
            loss_i = part.eval_loss(atom_target, atom_output_part, atom_mask, lightning_module=self, is_val=is_val,
                                    global_index=global_idx)
            self.log(val_or_train + "/" + part_name + "_tot", torch.sum(torch.flatten(loss_i)), on_step=True, on_epoch=True)
            atom_loss += loss_i

        for zi, part in enumerate(bond_parts):
            bond_mask = batch["bond_mask"][zi].to(device)
            part_name = part.__class__.__name__
            bond_target = dataset_helper.prepare_for_loss(bond_data, batch["bond_metadata"], part_name)
            bond_output_part = dataset_helper.prepare_for_loss(bond_output, batch["bond_metadata"], part_name)
            loss_i = part.eval_loss(bond_target, bond_output_part, bond_mask, lightning_module=self, is_val=is_val,
                           global_index=global_idx)
            self.log(val_or_train+"/"+part_name+"_tot",torch.sum(torch.flatten(loss_i)),on_step=True, on_epoch=True)
            bond_loss += loss_i

        atom_loss_tot = torch.sum(atom_loss, dim=(0, 1))
        bond_loss_tot = torch.sum(bond_loss, dim=(0, 1))
        total_loss = atom_loss_tot + bond_loss_tot
        return total_loss

    def training_step(self, batch, batch_idx):
        device = self.train_config.device
        atom_data = batch["atom_data"].to(device)
        bond_data = batch["bond_data"].to(device)

        outputs, latent = self(atom_data, bond_data)
        atom_output, bond_output = outputs[0], outputs[1]
        loss = self.loss_fn(atom_data, bond_data, atom_output, bond_output, batch, False, batch["global_index"])

        # Log the training loss on both step and epoch levels
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        device = self.train_config.device
        atom_data = batch["atom_data"].to(device)
        bond_data = batch["bond_data"].to(device)

        outputs, latent = self(atom_data, bond_data)
        atom_output, bond_output = outputs[0], outputs[1]
        loss = self.loss_fn(atom_data, bond_data, atom_output, bond_output, batch, True, batch["global_index"])

        self.log("val_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_config.LEARNING_RATE)

        # Setup the ramp-up scheduler and an additional cyclic scheduler.
        # Note: Lightning expects schedulers to be configured with 'interval' and (optionally) 'frequency'.
        ramp_scheduler = LinearRampUpScheduler(optimizer, self.ramp_up_steps, self.train_config.LEARNING_RATE)
        cyclic_scheduler = CyclicLR(
            optimizer,
            cycle_momentum=False,
            base_lr=0.2 * self.train_config.LEARNING_RATE,
            max_lr=self.train_config.LEARNING_RATE,
            step_size_up=400,
            step_size_down=400
        )

        # Returning multiple schedulers can be done by providing a list.
        # You might need to manage switching between these schedulers manually
        # or combine their functionality depending on your specific needs.
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ramp_scheduler,  # start with ramp scheduler
                "interval": "step",
            }
        }
