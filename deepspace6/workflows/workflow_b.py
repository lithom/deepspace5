import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from deepspace6 import ds6
from deepspace6.data.histogram_dataset import HistogramDataset
from deepspace6.data.molecule_dataset import InMemoryDataset
from deepspace6.models.basic_histogram_autoencoder import FullHistogramAutoencoder


from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        try:
            data = pickle.load(file)  # Load the entire pickle file
            print("Pickle file loaded successfully.")
            print(f"Data type: {type(data)}")  # Print the type of the loaded data
            return data
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            return None

if False:
    file_path = 'C:\dev\deepspace5\data\datasets\dataset_s32_a1.pickle'  # Replace with your pickle file path
    data = load_pickle(file_path)

    print("mkay")

if False:
    conformations = []
    for zi in range(16000):
        conformations.append( data['Samples'][zi]['coords3d'] )

    with open('datasets/conf_data_a.pkl', 'wb') as f:
        pickle.dump(conformations, f)

if True:
    data = load_pickle('datasets/conf_data_a.pkl')
    conformations = data[0:16000]
    print("mkay")



# Create dataset and dataloader
num_atoms = 32
num_bins = 64
#batch_size = 128
batch_size = 1024
device = "cuda"

if False:
    dataset = HistogramDataset(conformations, max_distance=32.0, num_bins=num_bins)
    dataset_inmem = InMemoryDataset(dataset)
    with open('datasets/hist_data_16k_b.pkl', 'wb') as f:
        pickle.dump(dataset_inmem, f)

if True:
    #dataset = HistogramDataset(conformations, max_distance=32.0, num_bins=num_bins)
    #dataset_inmem = load_pickle('datasets/hist_data_4k_a.pkl')
    #dataset_inmem = load_pickle('datasets/hist_data_16k_a.pkl')
    dataset_inmem = load_pickle('datasets/hist_data_16k_b.pkl')


#dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
dataloader = DataLoader(dataset_inmem, batch_size=batch_size, shuffle=True)

# Instantiate model
#model = FullHistogramAutoencoder(num_bins=num_bins, n2d=16, dim_latent=8192)
#model = FullHistogramAutoencoder(num_bins=num_bins, n2d=16, dim_latent=8192,n_layers=2)
#model = FullHistogramAutoencoder(num_bins=num_bins, n2d=8, dim_latent=8192,n_layers=1)
#model = FullHistogramAutoencoder(num_bins=num_bins, n2d=8, dim_latent=8192,n_layers=1)
model = FullHistogramAutoencoder(num_bins=num_bins, n2d=8, dim_latent=256,n_layers=1)
#model.load_state_dict(torch.load(f"{ds6.DS6_ROOT_DIR}/workflows/model_histo3d_latent8192_C_ckpt_1.ckpt", weights_only=True))
#model.load_state_dict(torch.load(f"{ds6.DS6_ROOT_DIR}/workflows/model_histo3d_latent8192_B_ckpt_99.ckpt", weights_only=True))
#model.load_state_dict(torch.load(f"{ds6.DS6_ROOT_DIR}/workflows/checkpoints/model_histo3d_latent8192_B_p2_ckpt_99.ckpt", weights_only=True))


model = model.to(device)  # Use GPU if available

count_parameters(model)

# Optimizer and loss function
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00025)
loss_function = nn.MSELoss()

# Training loop
num_epochs = 100
for epoch in range(0,100):
    model.train()
    epoch_loss = 0.0

    for batch in dataloader:
        batch_histo   = batch[0].to(device)  # Move batch to GPU if available
        batch_meanvar = batch[1].to(device)
        #!!! NORMALIZE !!!
        batch_meanvar = batch_meanvar / 32.0
        # Forward pass
        reconstructed_both, latent = model(batch_histo, batch_meanvar)
        reconstructed = reconstructed_both[0]
        reconstructed_meanvar = reconstructed_both[1]

        # Compute loss

        meanvar_loss = 10000*F.mse_loss(batch_meanvar, reconstructed_meanvar)

        # histogram_loss = torch.zeros(batch_size, num_atoms, num_atoms).to(device)
        batch_loss = torch.zeros(1,1).to(device)
        # Iterate over all 32x32 histograms
        for i in range(num_atoms-1):
            for j in range(1,num_atoms):
                # Select histograms for the atom pair (i, j)
                target_histogram = batch_histo[:, i, j, :]  # Shape: (batch_size, num_bins)
                target_histogram = torch.clamp(target_histogram, min=1e-6)
                target_histogram = target_histogram / target_histogram.sum(dim=-1, keepdim=True)  # Normalize target to probabilitie
                predicted_histogram = reconstructed[:, i, j, :]  # Shape: (batch_size, num_bins)

                # Normalize distributions
                # target_probs = F.softmax(target_histogram, dim=-1) + 1e-9
                # predicted_probs = F.softmax(predicted_histogram, dim=-1) + 1e-9

                # Compute KL divergence for the histogram
                #predicted_log_probs = predicted_histogram.log_softmax(dim=-1)
                predicted_log_probs = predicted_histogram
                #kl_loss = F.kl_div(predicted_log_probs, target_histogram, reduction='batchmean', log_target=False)
                kl_loss = F.mse_loss(predicted_histogram, target_histogram) + 16 * F.mse_loss(target_histogram*(target_histogram>1e-5), predicted_histogram*(target_histogram>1e-5))

                # Accumulate loss for the batch
                batch_loss += kl_loss
                #print(f"[{i},{j}] -> {kl_loss}")
                #histogram_loss[:,i,j] = kl_loss

        # Average the loss over 32x32 histograms
        # batch_loss = torch.sum(histogram_loss,(0,1,2))
        #batch_loss /= (32 * 32)

        print(f"lossMV: {meanvar_loss}")
        print(f"lossBatch: {batch_loss}")
        total_loss = meanvar_loss + batch_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_loss += batch_loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")
    torch.save(model.state_dict(), f"model_histo3d_latent8192_C_ckpt_{epoch}.ckpt")


print("done!")

