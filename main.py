import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=3072):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(max_seq_len).float()
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer('emb', emb)

    def forward(self, x):
        return self.emb[:x.shape[1], :]

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_embedding(q, k, rope_emb):
    # Reshape rope embeddings to match query and key shapes
    rope_emb = rope_emb.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]

    # Apply rotary embeddings
    q_rope = (q * rope_emb.cos()) + (rotate_half(q) * rope_emb.sin())
    k_rope = (k * rope_emb.cos()) + (rotate_half(k) * rope_emb.sin())
    return q_rope, k_rope

class RotaryMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Project queries, keys, and values and reshape
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Get rotary embeddings for the sequence length
        rope_emb = self.rope(x)  # [seq_len, head_dim]

        # Apply rotary embeddings to each head separately
        q_rotated = []
        k_rotated = []

        for head in range(self.num_heads):
            q_head = q[..., head, :]  # [batch_size, seq_len, head_dim]
            k_head = k[..., head, :]  # [batch_size, seq_len, head_dim]

            q_head_rot, k_head_rot = apply_rotary_embedding(q_head, k_head, rope_emb)
            q_rotated.append(q_head_rot)
            k_rotated.append(k_head_rot)

        # Stack the rotated heads back together
        q = torch.stack(q_rotated, dim=2)  # [batch_size, seq_len, num_heads, head_dim]
        k = torch.stack(k_rotated, dim=2)  # [batch_size, seq_len, num_heads, head_dim]

        # Reshape for attention computation
        q = q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)  # [batch_size, num_heads, seq_len, head_dim]
        out = out.transpose(1, 2)  # [batch_size, seq_len, num_heads, head_dim]
        out = out.reshape(batch_size, seq_len, self.embed_dim)

        return self.out_proj(out)

class RotaryTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = RotaryMultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.norm1(src)
        src2 = self.self_attn(src2)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

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
        super().__init__()
        self.layers = nn.ModuleList([
            RotaryTransformerEncoderLayer(embed_dim, num_heads)
            for _ in range(depth)
        ])
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        cond_embedding = self.fc(x[:, -1, :])
        return x, cond_embedding


class DiffusionTransformer(nn.Module):
    def __init__(self, latent_dim=512, num_tokens=16*16, diffusion_steps=10, num_heads=8, depth=6):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_tokens = num_tokens
        self.diffusion_steps = diffusion_steps

        self.layers = nn.ModuleList([
            RotaryTransformerEncoderLayer(latent_dim, num_heads)
            for _ in range(depth)
        ])
        self.fc_out = nn.Linear(latent_dim, latent_dim)

    def forward(self, cond_embedding):
        batch_size = cond_embedding.size(0)
        latent_tokens = torch.zeros(batch_size, self.num_tokens, self.latent_dim).to(cond_embedding.device)

        for step in range(self.diffusion_steps):
            # Apply transformer layers with RoPE
            for layer in self.layers:
                latent_tokens = layer(latent_tokens)

            # Apply conditional embedding
            latent_tokens[:, -1, :] = cond_embedding

            # Linear projection
            latent_tokens = self.fc_out(latent_tokens)

        decoded_input = latent_tokens.mean(dim=1)
        return decoded_input

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

        # Diffusion Transformer (Ne)
        decoded_input = self.diffusion_transformer(cond_embedding)  # Get reshaped latent

        # VAE Decoding (Si)
        output = self.decoder(decoded_input)  # Decode the reshaped latent

        return output, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss