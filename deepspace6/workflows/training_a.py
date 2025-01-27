import torch

from deepspace6.data.molecule_dataset import MoleculeDatasetHelper


def loss_fn(atom_data, bond_data, atom_output, bond_output, batch, dataset_helper: MoleculeDatasetHelper, device):
    atom_parts = dataset_helper.atom_embedding_parts
    bond_parts = dataset_helper.bond_embedding_parts

    # Compute losses for atom and bond embeddings
    atom_loss = torch.zeros(atom_data.size()[0], atom_data.size()[1]).to(device)
    bond_loss = torch.zeros(bond_data.size()[0], bond_data.size()[1]).to(device)

    for zi, part in enumerate(atom_parts):
        atom_mask = batch["atom_mask"][zi].to(device)
        part_name = part.__class__.__name__
        atom_target = dataset_helper.prepare_for_loss(atom_data, batch["atom_metadata"], part_name)
        atom_output_part = dataset_helper.prepare_for_loss(atom_output, batch["atom_metadata"], part_name)
        atom_loss += part.eval_loss(atom_target, atom_output_part, atom_mask)

    for zi, part in enumerate(bond_parts):
        bond_mask = batch["bond_mask"][zi].to(device)
        part_name = part.__class__.__name__
        bond_target = dataset_helper.prepare_for_loss(bond_data, batch["bond_metadata"], part_name)
        bond_output_part = dataset_helper.prepare_for_loss(bond_output, batch["bond_metadata"], part_name)
        bond_loss += part.eval_loss(bond_target, bond_output_part, bond_mask)

    # Total loss
    atom_loss_tot = torch.sum(atom_loss, dim=(0, 1))
    bond_loss_tot = torch.sum(bond_loss, dim=(0, 1))
    print(f" a={atom_loss_tot} b={bond_loss_tot} ")
    total_loss = atom_loss_tot + bond_loss_tot
    return total_loss

#def train_model(model, dataloader, dataset_helper: MoleculeDatasetHelper, dataloader_val, atom_parts, bond_parts, optimizer, num_epochs=50, device='cuda'):
def train_model(model, dataloader, dataset_helper: MoleculeDatasetHelper, dataloader_val,
                    optimizer, num_epochs=50, device='cuda'):
    """
    Train a model using the MoleculeDataset.
    :param model: PyTorch model to train.
    :param dataloader: dataloader for MoleculeDataset.
    :param dataset_helper: dataset helper for MoleculeDataset.
    :param atom_parts: List of atom embedding parts.
    :param bond_parts: List of bond embedding parts.
    :param optimizer: PyTorch optimizer.
    :param num_epochs: Number of training epochs.
    :param batch_size: Batch size for training.
    :param device: 'cpu' or 'cuda'.
    """
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)

    atom_parts = dataset_helper.atom_embedding_parts
    bond_parts = dataset_helper.bond_embedding_parts

    ckpt_counter = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch in dataloader:
            atom_data = batch["atom_data"].to(device)
            bond_data = batch["bond_data"].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs, latent = model(atom_data, bond_data)  # Example: Predicting atom embeddings
            atom_output = outputs[0]
            bond_output = outputs[1]
            total_loss = loss_fn(atom_data,bond_data,atom_output,bond_output,batch,dataset_helper,device)
            epoch_loss += total_loss.item()

            # Backward pass and optimization
            total_loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")

        # Validation phase
        # val_loss, avg_forward_time = validate(model, dataloader_val,dataset_val,atom_parts,bond_parts, device=device)
        val_loss, avg_forward_time = validate(model, dataloader_val, dataset_helper, device=device)

        #print(f"Epoch {epoch + 1}/{settings.NUM_EPOCHS} - Validation Loss: {val_loss:.4f}")
        print(f"Average Forward Pass Time: {avg_forward_time:.4f} seconds")

        # Save the model checkpoint if validation loss improves
        if val_loss < best_val_loss:
            print("Validation loss improved, saving model...")
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"model_ckpt_{ckpt_counter}.ckpt")
            ckpt_counter = ckpt_counter+1


def validate(model, val_loader, dataset_helper: MoleculeDatasetHelper, device='cuda'):
    """
    Validate the model and measure forward pass time.

    :param model: The PyTorch model.
    :param val_loader: DataLoader for validation data.
    :param loss_fn: Loss function.
    :param settings: Settings object with validation configurations.
    :return: Tuple (validation loss, average forward pass time).
    """

    model.eval()
    model.to(device)
    val_loss = 0.0
    total_time = 0.0

    with torch.no_grad():
        for batch in val_loader:
            atom_data = batch["atom_data"].to(device)
            bond_data = batch["bond_data"].to(device)
            # Forward pass

            # Measure forward pass time
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)

            start_time.record()
            outputs, latent = model(atom_data, bond_data)  # Example: Predicting atom embeddings
            end_time.record()

            atom_output = outputs[0]
            bond_output = outputs[1]

            torch.cuda.synchronize()
            total_time += start_time.elapsed_time(end_time) / 1000.0  # Convert ms to seconds

            total_loss = loss_fn(atom_data, bond_data, atom_output, bond_output, batch, dataset_helper, device)
            val_loss += total_loss.item()

    val_loss /= len(val_loader.dataset)
    avg_forward_time = total_time / len(val_loader)

    return val_loss, avg_forward_time
