import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the LipType model
LipType_model = load_model('LipType_model.h5')  # Replace 'LipType_model.h5' with the path to your LipType model file

# Function to preprocess video frames
def preprocess_frame(frame):
    # Perform any necessary preprocessing (e.g., resizing, normalization)
    # Resize the frame to match the input size of the LipType model
    resized_frame = cv2.resize(frame, (100, 50))
    # Normalize pixel values
    normalized_frame = resized_frame / 255.0
    return normalized_frame

# Function to extract frames from videos using the specified formula
def extract_frames(video_path, num_frames_formula):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    for x in range(1, total_frames + 1):
        frame_idx = int(num_frames_formula(x))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames

# Function to predict transcription for a set of frames
def predict_transcription(frames):
    transcriptions = []
    for frame in frames:
        preprocessed_frame = preprocess_frame(frame)
        preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)  # Add batch dimension
        transcription = LipType_model.predict(preprocessed_frame)
        transcriptions.append(transcription)
    return transcriptions

# Function to update the text file with transcriptions
def update_text_file(video_file, transcriptions, output_file):
    with open(output_file, 'a') as f:
        f.write(f"Video: {video_file}\n")
        for transcription in transcriptions:
            f.write(transcription + '\n')
        f.write('\n')

# Main function to process videos in a folder
def process_videos_in_folder(folder_path, output_file):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                frames = extract_frames(video_path, lambda x: x + 5)
                transcriptions = predict_transcription(frames)
                update_text_file(file, transcriptions, output_file)

# Example usage
folder_path = 'videos_folder'  # Replace 'videos_folder' with the path to your folder containing videos
output_file = 'transcriptions.txt'  # Output file to save transcriptions
process_videos_in_folder(folder_path, output_file)
