import torch
import torch.nn.functional as F

from dataloader import VideoDataset
from main import WorldModel, vae_loss

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# PHASE 1: Train the VAE (Se and Si)
def train_vae(model, dataloader, optimizer, num_epochs=10, device='cuda'):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, frames in enumerate(dataloader):
            # Reshape frames to (batch_size * clip_len, channels, height, width)
            frames = frames.view(-1, frames.shape[2], frames.shape[3], frames.shape[4]).to(device) 
            
            optimizer.zero_grad()

            # Forward pass through ONLY VAE encoder and decoder
            mu, logvar = model.encoder(frames)
            z = model.reparameterize(mu, logvar) # Crucial for VAE training
            recon = model.decoder(z)
            
            # Reshape recon to match the original frames shape for loss calculation
            recon = recon.view(frames.shape[0], -1, recon.shape[1], recon.shape[2], recon.shape[3])

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
        for batch_idx, (clip, target_frame) in enumerate(dataloader): # Unpack the tuple
            clip = clip.to(device) # Move sequence to device
            target_frame = target_frame.to(device) # Move target frame to device
            optimizer.zero_grad()
            
            # Reshape clip to (batch_size * clip_len, channels, height, width) before passing to the model
            clip_reshaped = clip.view(-1, clip.shape[2], clip.shape[3], clip.shape[4])

            recon, mu, logvar = model(clip_reshaped)  # Pass the reshaped clip to the model
            loss = F.mse_loss(recon, target_frame) # Calculate loss against the target frame

            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item()}')

    print('Phase 2: Transformer and Diffusion Transformer training with frozen VAE completed!')

# PHASE 3: Train the whole world model with next-frame prediction
def train_full_model(model, dataloader, optimizer, num_epochs=10, device='cuda'):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (clip, target_frame) in enumerate(dataloader): # Unpack the tuple
            clip = clip.to(device) # Move sequence to device
            target_frame = target_frame.to(device) # Move target frame to device
            optimizer.zero_grad()
            
            # Reshape clip to (batch_size * clip_len, channels, height, width) before passing to the model
            clip_reshaped = clip.view(-1, clip.shape[2], clip.shape[3], clip.shape[4])

            recon, mu, logvar = model(clip_reshaped)  # Pass the reshaped clip to the model
            loss = F.mse_loss(recon, target_frame) # Calculate loss against the target frame

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


# Dataloaders and dataset assumed to be already defined
# `dataloader` is your DataLoader object

video_path = "video.mp4"  # Provide the local path to your video file

video_transform = transforms.Compose([
    transforms.Lambda(lambda x: x / 255.0),  # Normalize to [0, 1]
    transforms.ConvertImageDtype(torch.float)
])

# Create dataloaders for each training phase

# Phase 1: VAE Training
vae_dataset = VideoDataset(video_path, phase=1, clip_len=48, stride=12, frame_size=(256, 256), transform=video_transform)
vae_dataloader = DataLoader(vae_dataset, batch_size=1, shuffle=True, num_workers=4)

# Phase 2 & 3: Transformer and Full Model Training
transformer_dataset = VideoDataset(video_path, phase=2, clip_len=12, stride=6, frame_size=(256, 256), transform=video_transform)  # Use phase=2 or 3
transformer_dataloader = DataLoader(transformer_dataset, batch_size=1, shuffle=False, num_workers=4)  # Don't shuffle for next-frame prediction

# PHASE 1: Train VAE
train_vae(model, vae_dataloader, optimizer_vae, num_epochs=10, device=device)

# PHASE 2: Train transformer and diffusion transformer with VAE frozen
train_transformer_with_frozen_vae(model, transformer_dataloader, optimizer_transformer, num_epochs=10, device=device)

# PHASE 3: Fine-tune the full world model
train_full_model(model, transformer_dataloader, optimizer_full, num_epochs=5, device=device)