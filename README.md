# MELDER

This code presents the implementation of a real-time silent speech recognizer, as outlined in the paper 'MELDER: The Design and Evaluation of a Real-time Silent Speech Recognizer for Mobile Devices' authored by by [Laxmi Pandey](https://lpandey21.github.io/) and [Ahmed Sabbir Arif](https://www.asarif.com/).

**Requirements**
* Python 3.x
* OpenCV (cv2)
* NumPy
* TensorFlow (for loading the pre-trained model)
  
**Setup**
* Ensure that you have all the required dependencies installed.
* Place your videos in a folder and specify the folder path in the folder_path variable.
* Replace 'LipType_model.h5' with the path to your pre-trained lipreading model file.
* Define the output file name in the output_file variable.
  
**Usage**
```
python transcribe.py

//The script will process each video in the specified folder, extract frames, predict transcriptions, and append them to the output text file.
```
**Functions**
* preprocess_frame(frame): Preprocesses a single frame by resizing it and normalizing pixel values.
* extract_frames(video_path, num_frames_formula): Extracts frames from a video using a specified formula to determine which frames to extract.
* predict_transcription(frames): Predicts transcriptions for a set of frames using the pre-trained LipType model.
* update_text_file(video_file, transcriptions, output_file): Updates the text file with transcriptions for a video.
* process_videos_in_folder(folder_path, output_file): Processes all videos in a folder, extracting frames, predicting transcriptions, and updating the output file.


**Note**
* The lipreading model file needs to be in the correct path or replaced with the appropriate path to your model file.
* Ensure that the videos in the specified folder are in a compatible format (e.g., .mp4).
* Adjust any parameters or configurations according to your requirements.
* We're unable to share the code intended for the reviewer channel utilizing the COCA dataset. This is because the dataset isn't accessible to the public, and we're unable to distribute it due to copyright limitations.
