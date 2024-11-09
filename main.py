import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torchvision.transforms as transforms
import os
import glob

from torch.utils.data import Dataset, DataLoader

class VideoDatasetWithWindow(Dataset):
    def __init__(self, video_dir, clip_len=240, stride=1, frame_size=(1024, 1024), transform=None):
        """
        Args:
            video_dir (str): Directory where video files are located.
            clip_len (int): Number of frames per clip (240 frames = 10 seconds at 24fps).
            stride (int): Number of frames to move the window (e.g., 1 frame = fully overlapping, 240 = non-overlapping).
            frame_size (tuple): Size of each frame (default 1024x1024).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.video_dir = video_dir
        self.clip_len = clip_len
        self.stride = stride
        self.frame_size = frame_size
        self.transform = transform

        # Get the list of video file paths
        self.video_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))

    def __len__(self):
        # Calculate total number of windows across all videos
        total_windows = 0
        for video_file in self.video_files:
            total_frames = self._get_total_frames(video_file)
            total_windows += (total_frames - self.clip_len) // self.stride + 1
        return total_windows

    def _get_total_frames(self, video_file):
        # Use OpenCV to get the total number of frames in the video
        cap = cv2.VideoCapture(video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total_frames

    def __getitem__(self, idx):
        # Find which video the index corresponds to and where the window starts
        video_idx, window_start_frame = self._find_video_and_frame(idx)

        # Load the corresponding video
        video_file = self.video_files[video_idx]
        video_frames = self._load_video_frames(video_file)

        # Extract the window of frames for the current index
        clip = video_frames[window_start_frame: window_start_frame + self.clip_len]

        # Resize the frames to the desired frame size
        clip = torch.stack([transforms.functional.resize(frame, self.frame_size) for frame in clip])

        # Optionally apply transformations (e.g., normalization)
        if self.transform:
            clip = self.transform(clip)

        # Return the windowed clip
        return clip

    def _find_video_and_frame(self, idx):
        """
        Find the video and the start frame for the window based on the global index.
        """
        total_frames_so_far = 0

        for video_idx, video_file in enumerate(self.video_files):
            total_frames_in_video = self._get_total_frames(video_file)
            total_windows_in_video = (total_frames_in_video - self.clip_len) // self.stride + 1

            if idx < total_windows_in_video:
                # The index belongs to this video
                window_start_frame = idx * self.stride
                return video_idx, window_start_frame
            else:
                # Skip this video and continue searching
                idx -= total_windows_in_video

        raise IndexError("Index out of range in the dataset")

    def _load_video_frames(self, video_file):
        """
        Load all the frames of the video using OpenCV.
        """
        cap = cv2.VideoCapture(video_file)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
            frames.append(frame)
        
        cap.release()
        return frames
    
# Example normalization transform (if needed)
video_transform = transforms.Compose([
    transforms.Lambda(lambda x: x / 255.0),  # Normalize to [0, 1]
])

# Create the dataset with sliding window
video_dataset = VideoDatasetWithWindow(video_dir="/path/to/your/video/files", 
                                       clip_len=240,  # 10 seconds of video at 24fps
                                       stride=120,    # Window stride (e.g., 120 = 5-second overlap)
                                       frame_size=(1024, 1024),
                                       transform=video_transform)

# Create the DataLoader
video_loader = DataLoader(video_dataset, batch_size=4, shuffle=True, num_workers=4)

# VAE Encoder (Se)
class VAEEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=512, latent_dim=512):
        super(VAEEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)  # 1024 -> 512
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 512 -> 256
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 256 -> 128
        self.conv4 = nn.Conv2d(256, hidden_dim, kernel_size=4, stride=2, padding=1)  # 128 -> 64
        self.fc_mu = nn.Linear(hidden_dim * 16 * 16, latent_dim)  # compressed to latent space
        self.fc_logvar = nn.Linear(hidden_dim * 16 * 16, latent_dim)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)  # flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# VAE Decoder (Si)
class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=512, hidden_dim=512, out_channels=3):
        super(VAEDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim * 16 * 16)
        self.conv1 = nn.ConvTranspose2d(hidden_dim, 256, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 16, 16)  # reshape back to image size
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))  # output image
        return x

class TemporalTransformer(nn.Module):
    def __init__(self, seq_len=240, embed_dim=512, num_heads=8, depth=6):
        super(TemporalTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.fc = nn.Linear(embed_dim, embed_dim)  # to process the last hidden state

    def forward(self, x):
        # Input shape: (batch_size, seq_len, embed_dim)
        x = self.transformer(x)
        # Take the last hidden state as the conditional embedding
        cond_embedding = self.fc(x[:, -1, :])
        return x, cond_embedding

class DiffusionTransformer(nn.Module):
    def __init__(self, latent_dim=512, num_tokens=16*16, diffusion_steps=10, num_heads=8, depth=6):
        super(DiffusionTransformer, self).__init__()
        self.latent_dim = latent_dim
        self.num_tokens = num_tokens  # 16x16 tokens for one frame
        self.diffusion_steps = diffusion_steps
        
        # Define a transformer block
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Linear projection to output full frame (16x16 tokens)
        self.fc_out = nn.Linear(latent_dim, latent_dim)

    def forward(self, cond_embedding):
        batch_size = cond_embedding.size(0)
        
        # Start with zero tensor of shape (batch_size, num_tokens, latent_dim)
        latent_tokens = torch.zeros(batch_size, self.num_tokens, self.latent_dim).to(cond_embedding.device)
        
        for step in range(self.diffusion_steps):
            # Apply transformer over all tokens
            latent_tokens = self.transformer(latent_tokens)
            
            # Use conditional embedding to guide the last token(s)
            # Here, instead of applying it to one token, we apply it over all tokens
            latent_tokens[:, -1, :] = cond_embedding  # Apply cond embedding to the last token
            
            # Linear projection over all tokens
            latent_tokens = self.fc_out(latent_tokens)
        
        # Return the predicted latent tokens (16x16 tokens for one frame)
        return latent_tokens

class WorldModel(nn.Module):
    def __init__(self):
        super(WorldModel, self).__init__()
        self.encoder = VAEEncoder()
        self.temporal_transformer = TemporalTransformer()
        self.diffusion_transformer = DiffusionTransformer()
        self.decoder = VAEDecoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # VAE Encoding (Se)
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        
        # Reshape z to match the expected transformer input (batch_size, seq_len, embed_dim)
        z_seq = z.view(x.size(0), -1, 512)
        
        # Temporal Transformer (Ni)
        hidden_states, cond_embedding = self.temporal_transformer(z_seq)
        
        # Diffusion Transformer (Ne) now predicts a full frame's latent tokens
        latent_tokens = self.diffusion_transformer(cond_embedding)
        
        # VAE Decoding (Si) reconstructs the frame from latent tokens
        output = self.decoder(latent_tokens)
        
        return output, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss



# Example usage:
"""
model = WorldModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    for batch in video_loader:
        batch = batch.to(device)  # Move video batch to the correct device (GPU or CPU)

        optimizer.zero_grad()

        # Forward pass (batch is of shape [batch_size, 240, 3, 1024, 1024])
        output, mu, logvar = model(batch)
        
        # Assuming VAE loss function from earlier
        loss = vae_loss(output, batch, mu, logvar)
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item()}")
"""
