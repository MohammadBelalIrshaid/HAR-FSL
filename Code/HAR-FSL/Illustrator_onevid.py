import os
import cv2

def illustrate_and_save_videos(folder_path, output_video_name="Dataset_Walking.avi"):
    # Ensure the provided path exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")
    
    # List video files in the folder
    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.avi', '.mp4', '.mkv'))]
    
    if len(video_files) != 3:
        raise ValueError("The folder must contain exactly 3 DVS videos.")
    
    # Sort videos alphabetically or as needed
    video_files.sort()
    video_paths = [os.path.join(folder_path, video_file) for video_file in video_files]
    
    # Open the first video to get frame properties
    sample_cap = cv2.VideoCapture(video_paths[0])
    frame_width = int(sample_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(sample_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(sample_cap.get(cv2.CAP_PROP_FPS))
    sample_cap.release()
    
    # Initialize video writer
    output_path = os.path.join(folder_path, output_video_name)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use XVID codec
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))
    
    print(f"Saving illustrated video to: {output_path}")
    
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_file}")
        
        # Extract label from the video file name (before the first underscore or full name if not delimited)
        label = video_file.split('_')[0] if '_' in video_file else video_file.split('.')[0]
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video
            
            # Add label as text on the frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, label, (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Write the frame to the output video
            out.write(frame)
        
        cap.release()
    
    # Release video writer
    out.release()
    print("Video saved successfully.")

# Example usage:
# Replace 'your_folder_path' with the actual path to your folder
folder_path = r'C:\Users\Mohammad Belal\Desktop\Tarnini_extractions\Walking'
illustrate_and_save_videos(folder_path)
