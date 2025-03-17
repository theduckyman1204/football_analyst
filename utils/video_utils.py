import cv2

def read_video(video_path): 
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True: 
        ret,frame = cap.read()#ret is True if the reading is successsful and is False if not 
        if not ret: 
            break
        frames.append(frame)
    return frames 
def save_video(output_video_frames, output_video_path):
    if not output_video_frames:
        print("No frames to save.")
        return

    height, width, _ = output_video_frames[0].shape
    fourcc = int(cv2.VideoWriter.fourcc(*"XVID"))
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))

    for frame in output_video_frames:
        out.write(frame)

    out.release()
    print(f"Video saved to {output_video_path}")



