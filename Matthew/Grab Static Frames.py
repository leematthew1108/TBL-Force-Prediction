import cv2
import numpy as np
import os


def grab_static_frames(video_file, output_dir):
    # Initialize VideoCapture object
    vidPath = cv2.VideoCapture(video_file)

    # Check if the video file was opened successfully
    if not vidPath.isOpened():
        print("Error: Unable to open the video file.")
        return

    # Initialize variables
    frames_with_audio_spike = []
    frame_count = 0

    # Load the audio data from the video file
    audio = AudioSegment.from_file(video_file)
    audio = audio.set_channels(1)  # Convert to mono
    audio_data = np.array(audio.get_array_of_samples())
    fs = 48000  # Audio Sampling Frequency

    # Initialize variables for tracking audio levels
    audio_levels = []
    window_size = 100  # Window size is adjustable (need to test)
    threshold_factor = 4  # Factor is adjustable (need to test)

    # Find frames with local audio spikes
    while True:
        ret, frame = vidPath.read()
        if not ret:
            break

        frame_count += 1

        # Calculate the audio level of the current frame
        frame_audio_level = max(
            abs(sample)
            for sample in audio_data[
                frame_count
                * fs
                // vidPath.get(cv2.CAP_PROP_FPS) : (frame_count + 1)
                * fs
                // vidPath.get(cv2.CAP_PROP_FPS)
            ]
        )

        # Add the audio level to the list
        audio_levels.append(frame_audio_level)

        # Maintain a sliding window of audio levels
        if len(audio_levels) > window_size:
            audio_levels.pop(0)

        # Check for local maximum
        if len(audio_levels) == window_size and frame_count > window_size:
            if audio_levels[window_size // 2] > max(
                audio_levels[: window_size // 2]
            ) * threshold_factor and audio_levels[window_size // 2] > max(
                audio_levels[window_size // 2 + 1 :]
            ):
                frames_with_audio_spike.append((frame_count, frame))

                # Save these frames as images if needed
                # cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count}.jpg"), frame)

                # After 3 frames have been added, I can break the loop
                if len(frames_with_audio_spike) == 3:
                    break

    return frames_with_audio_spike


# Example usage:
# frames = grab_static_frames("your_video_file.mp4", "output_directory")
# for frame_num, frame in frames:
#     cv2.imshow(f"Frame {frame_num}", frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
