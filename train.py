import torch
import torch.nn.functional as F

from main import WorldModel, vae_loss

# PHASE 1: Train the VAE (Se and Si)
def train_vae(model, dataloader, optimizer, num_epochs=10, device='cuda'):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, frames in enumerate(dataloader):
            frames = frames.to(device)
            optimizer.zero_grad()

            # Forward pass through ONLY VAE encoder and decoder
            mu, logvar = model.encoder(frames)
            z = model.reparameterize(mu, logvar) # Crucial for VAE training
            recon = model.decoder(z)

            loss = vae_loss(recon, frames, mu, logvar)

            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item()}')
                
    print('Phase 1: VAE training completed!')

# PHASE 2: Train Transformer (Ni) and Diffusion Transformer (Ne) with VAE Frozen
def train_transformer_with_frozen_vae(model, dataloader, optimizer, num_epochs=10, device='cuda'):
    # Freeze VAE
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = False

    model.train()
    for epoch in range(num_epochs):
        for batch_idx, frames in enumerate(dataloader):
            frames = frames.to(device)
            optimizer.zero_grad()

            recon, mu, logvar = model(frames)  # Use the model's forward method
            loss = F.mse_loss(recon, frames[:, -1]) # Reconstruction loss on the last frame

            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item()}')

    print('Phase 2: Transformer and Diffusion Transformer training with frozen VAE completed!')

# PHASE 3: Train the whole world model with next-frame prediction
def train_full_model(model, dataloader, optimizer, num_epochs=10, device='cuda'):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, frames in enumerate(dataloader):
            frames = frames.to(device)
            optimizer.zero_grad()

            recon, mu, logvar = model(frames)  # Use the model's forward method
            loss = F.mse_loss(recon, frames[:, -1]) # Reconstruction loss on the last frame

            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item()}')

    print('Phase 3: Full world model training completed!')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = WorldModel().to(device)

# Example optimizer setup
vae_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
optimizer_vae = torch.optim.Adam(vae_params, lr=1e-4)

transformer_params = list(model.temporal_transformer.parameters()) + list(model.diffusion_transformer.parameters())
optimizer_transformer = torch.optim.Adam(transformer_params, lr=1e-4)

full_model_params = list(model.parameters())  # This includes VAE, temporal transformer, and diffusion transformer
optimizer_full = torch.optim.Adam(full_model_params, lr=1e-5)

"""
# Dataloaders and dataset assumed to be already defined
# `dataloader` is your DataLoader object

# PHASE 1: Train VAE
train_vae(model, dataloader, optimizer_vae, num_epochs=10, device=device)

# PHASE 2: Train transformer and diffusion transformer with VAE frozen
train_transformer_with_frozen_vae(model, dataloader, optimizer_transformer, num_epochs=10, device=device)

# PHASE 3: Fine-tune the full world model
train_full_model(model, dataloader, optimizer_full, num_epochs=5, device=device)
"""