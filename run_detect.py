"""Quick diagnostic: capture one camera frame and run `get_emotion`.
Saves the captured frame to `debug_frame.jpg` and prints backend + result.
"""
import time
import cv2
from src import detect_emotion


def main():
    print("Using backend:", detect_emotion.USE_BACKEND)
    # try to capture a single frame from the default camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to open camera (index 0).")
        return
    time.sleep(1.0)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("Failed to read frame from camera.")
        return
    # mirror to match app
    frame = cv2.flip(frame, 1)
    cv2.imwrite("debug_frame.jpg", frame)
    print("Saved debug_frame.jpg (check it to ensure the camera captured correctly)")
    try:
        emotion, conf = detect_emotion.get_emotion(frame)
        print("get_emotion ->", emotion, conf)
    except Exception as e:
        print("get_emotion raised:", type(e).__name__, e)


if __name__ == "__main__":
    main()
