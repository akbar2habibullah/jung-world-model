import torch
import cv2
import torchvision.transforms as transforms
import os

from torch.utils.data import Dataset, DataLoader

class VideoDataset(Dataset):
    def __init__(self, video_path, phase, clip_len=240, stride=120, frame_size=(256, 256), transform=None):
        self.video_path = video_path
        self.clip_len = clip_len
        self.stride = stride
        self.frame_size = frame_size
        self.transform = transform
        self.phase = phase

        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        self.total_frames = self._get_total_frames(self.video_path)

    def _get_total_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total_frames
        
    def __len__(self):
        if self.phase == 1:  # For VAE training, use the entire video
            # Ensure the result is non-negative by using max(0, ...)
            length = (self.total_frames - self.clip_len) // self.stride + 1
            # Print length to check if it's 0
            print(f"Dataset length (phase 1): {length}")  # Add this line for debugging
            return max(0, length)  # Return 0 if length is negative
        else:  # For transformer and full model training, predict the last frame
            # Ensure the result is non-negative by using max(0, ...)
            length = self.total_frames - self.clip_len  # Number of possible starting frames
            # Print length to check if it's 0
            print(f"Dataset length (phase 2/3): {length}")  # Add this line for debugging
            return max(0, length)  # Return 0 if length is negative


    def __getitem__(self, idx):

        cap = cv2.VideoCapture(self.video_path)

        if self.phase == 1:  # VAE training (return clips)
            start_frame = idx * self.stride
            clip = []
            for i in range(self.clip_len):
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i)
                ret, frame = cap.read()
                if not ret:
                    frame = torch.zeros(3, *self.frame_size, dtype=torch.uint8)
                else:
                    frame = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
                    frame = transforms.functional.resize(frame, self.frame_size)
                clip.append(frame)
            clip = torch.stack(clip)

        else: # Transformer and full model training (return sequence and target last frame)
            start_frame = idx
            clip = [] #Sequence
            for i in range(self.clip_len): # Sequence frames
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i)
                ret, frame = cap.read()
                if not ret:
                    frame = torch.zeros(3, *self.frame_size, dtype=torch.uint8)
                else:
                    frame = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
                    frame = transforms.functional.resize(frame, self.frame_size)
                clip.append(frame)

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + self.clip_len) # Target frame
            ret, target_frame = cap.read()
            if not ret:
                target_frame = torch.zeros(3, *self.frame_size, dtype=torch.uint8)
            else:
                target_frame = torch.from_numpy(cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
                target_frame = transforms.functional.resize(target_frame, self.frame_size)
            clip = torch.stack(clip)

        cap.release()


        if self.transform:
            if self.phase == 1:
                clip = self.transform(clip)
            else:
                clip = self.transform(clip)
                target_frame = self.transform(target_frame)


        if self.phase == 1:
          return clip
        else:
          return clip, target_frame

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