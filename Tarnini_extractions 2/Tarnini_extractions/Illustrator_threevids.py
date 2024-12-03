import os
import cv2

def illustrate_videos_side_by_side(folder_path, output_video_name="Dataset_Jumping_three.avi"):
    # Ensure the provided path exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")
    
    # Get folder name as the title
    folder_name = os.path.basename(os.path.normpath(folder_path))
    
    # List video files in the folder
    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.avi', '.mp4', '.mkv'))]
    
    if len(video_files) != 3:
        raise ValueError("The folder must contain exactly 3 DVS videos.")
    
    # Sort videos alphabetically or as needed
    video_files.sort()
    video_paths = [os.path.join(folder_path, video_file) for video_file in video_files]
    
    # Open the videos
    caps = [cv2.VideoCapture(video_path) for video_path in video_paths]
    
    # Check if all videos are opened successfully
    if not all(cap.isOpened() for cap in caps):
        raise ValueError("One or more videos could not be opened.")
    
    # Get the frame properties from the first video
    frame_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(caps[0].get(cv2.CAP_PROP_FPS))
    
    # Define the output video size (three times the width to fit three videos side by side)
    combined_width = frame_width * 3
    combined_height = frame_height
    
    # Initialize video writer
    output_path = os.path.join(folder_path, output_video_name)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use XVID codec
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (combined_width, combined_height))
    
    print(f"Saving side-by-side video to: {output_path}")
    
    # Labels extracted from filenames
    labels = [
        video_file.split('_')[0] if '_' in video_file else video_file.split('.')[0]
        for video_file in video_files
    ]
    
    while True:
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                frames.append(None)
            else:
                frames.append(frame)
        
        # Stop if any video ends
        if any(frame is None for frame in frames):
            break
        
        # Resize frames if necessary (ensuring they have the same size)
        frames = [
            cv2.resize(frame, (frame_width, frame_height)) if frame is not None else None
            for frame in frames
        ]
        
        # Concatenate frames horizontally
        combined_frame = cv2.hconcat(frames)
        
        # Add overlay text for labels and folder title
        cv2.putText(combined_frame, f"{folder_name}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  # Folder name in white
        for i, label in enumerate(labels):
            x_offset = i * frame_width + 20
            y_offset = frame_height - 30
            cv2.putText(combined_frame, f"{label}", (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)  # Labels in blue
        
        # Write the combined frame to the output video
        out.write(combined_frame)
    
    # Release resources
    for cap in caps:
        cap.release()
    out.release()
    
    print("Video saved successfully.")

# Example usage:
# Replace 'your_folder_path' with the actual path to your folder
folder_path = r'C:\Users\Mohammad Belal\Desktop\Tarnini_extractions\Jumping'
illustrate_videos_side_by_side(folder_path)
