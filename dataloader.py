import torch
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

        # Load the corresponding video CAPTURE OBJECT (not all frames at once!)
        cap = cv2.VideoCapture(self.video_files[video_idx])

        clip = []
        for i in range(self.clip_len):
            cap.set(cv2.CAP_PROP_POS_FRAMES, window_start_frame + i)
            ret, frame = cap.read()
            if not ret:
                # Handle the case where the video ends before the clip is complete
                # You might want to pad with zeros or raise an error.
                frame = torch.zeros(3, *self.frame_size, dtype=torch.uint8)  # Example: Pad with black frames
            else:
                frame = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
                frame = transforms.functional.resize(frame, self.frame_size)
            clip.append(frame)
        cap.release()

        clip = torch.stack(clip)

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