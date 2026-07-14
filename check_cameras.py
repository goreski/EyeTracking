import cv2

def list_available_cameras(max_tested=5):
    """Checks and lists available camera indexes."""
    available_cameras = []
    for index in range(max_tested):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            # Try to grab a frame to confirm it works
            ret, frame = cap.read()
            if ret:
                print(f"Camera found at index: {index}")
                available_cameras.append(index)
        cap.release()
    
    if not available_cameras:
        print("No functional cameras detected.")
    return available_cameras

if __name__ == "__main__":
    list_available_cameras()