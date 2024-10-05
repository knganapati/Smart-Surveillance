
import cv2
import time
import os

# Initialize camera
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide a video file path

# Initialize variables
output_folder = 'recorded_videos/'  # Output folder for recorded videos
combined_output_folder = 'todays_record/'  # Folder for combined video
recording = False
out = None
previous_frame = None  # Initialize previous frame
motion_detected = False  # Initialize motion detection flag
no_motion_count = 0  # Initialize no motion counter
recorded_videos = []  # List to store filenames of recorded videos
combine_time = "00:34"  # Time to combine videos (HH:MM format)

# Ensure the output folders exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(combined_output_folder):
    os.makedirs(combined_output_folder)

# Define motion detection function
def detect_motion(current_frame, previous_frame):
    gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    gray_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    frame_diff = cv2.absdiff(gray_current, gray_previous)
    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            motion_detected = True
            break

    return motion_detected

# Main loop
while True:
    ret, frame = cap.read()

    if not ret:
        break

    if previous_frame is not None:
        motion_detected = detect_motion(frame, previous_frame)

    if motion_detected and not recording:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = output_folder + f'recorded_video_{timestamp}.avi'

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_file, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
        recording = True
        print(f"Recording started ({output_file})...")

    if not motion_detected and recording:
        no_motion_count += 1

        if no_motion_count > 40:  # Assuming 20 frames per second, 40 frames = 2 seconds
            out.release()
            recording = False
            print(f"Recording stopped ({output_file}).")
            recorded_videos.append(output_file)
            no_motion_count = 0

    else:
        no_motion_count = 0

    if recording:
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, current_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        out.write(frame)

    cv2.imshow('Live Feed', frame)

    previous_frame = frame.copy()

    # Check if it's time to combine videos
    current_hour_minute = time.strftime("%H:%M")
    if current_hour_minute == combine_time and len(recorded_videos) > 0:
        print(f"Combining {len(recorded_videos)} recorded videos at {combine_time}...")

        # Combine recorded videos into a single video
        first_video = cv2.VideoCapture(recorded_videos[0])
        frame_width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        first_video.release()

        combined_output_file = combined_output_folder + f'combined_video_{time.strftime("%Y%m%d-%H%M%S")}.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        combined_out = cv2.VideoWriter(combined_output_file, fourcc, 20.0, (frame_width, frame_height))

        for video_file in recorded_videos:
            cap = cv2.VideoCapture(video_file)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                combined_out.write(frame)
            cap.release()

        combined_out.release()
        print(f"Combined video saved as {combined_output_file}")

        # Clear recorded videos list after combining
        recorded_videos = []

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if recording:
    out.release()
cv2.destroyAllWindows()
