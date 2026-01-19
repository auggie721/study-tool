import time
import cv2
import pyautogui
from ultralytics import YOLO

# --- Settings ---
CONF_THRESH = 0.45          # detection confidence threshold
HOLD_SECONDS = 2        # phone must be detected continuously
COOLDOWN_SECONDS = 0       # cooldown between popups
CAMERA_INDEX = 0            # default webcam

# Performance / resource tuning
FRAME_WIDTH = 320          # capture width (keep small)
FRAME_HEIGHT = 240         # capture height
# Run inference less frequently to reduce CPU usage
INFERENCE_EVERY_N_FRAMES = 60
# Run the model on a smaller copy of the frame and scale boxes back
INFERENCE_WIDTH = 160
INFERENCE_HEIGHT = 120
# Disable display (set True to hide the OpenCV window and reduce GPU/CPU used by GUI)
SKIP_DISPLAY = False

def main():
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    phone_present_since = None
    last_popup_time = 0

    frame_count = 0
    last_results = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_count += 1

        # Run YOLO only every N frames on a downscaled copy
        if frame_count % INFERENCE_EVERY_N_FRAMES == 0:
            small = cv2.resize(frame, (INFERENCE_WIDTH, INFERENCE_HEIGHT))
            last_results = model(small, verbose=False)[0]

        phone_detected = False

        # Use last known detection results
        if last_results is not None and last_results.boxes is not None:
            for box in last_results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # COCO class 67 = cell phone
                if cls_id == 67 and conf >= CONF_THRESH:
                    phone_detected = True

                    # Draw bounding box (debug). Results were produced on the
                    # downscaled frame so we scale coordinates back to the
                    # original frame size for display.
                    x1_s, y1_s, x2_s, y2_s = map(float, box.xyxy[0])
                    scale_x = frame.shape[1] / INFERENCE_WIDTH
                    scale_y = frame.shape[0] / INFERENCE_HEIGHT
                    x1 = int(x1_s * scale_x)
                    y1 = int(y1_s * scale_y)
                    x2 = int(x2_s * scale_x)
                    y2 = int(y2_s * scale_y)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"PHONE {conf:.2f}",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                    break  # no need to check other boxes

        now = time.time()

        # Require sustained detection
        if phone_detected:
            if phone_present_since is None:
                phone_present_since = now
            elif (now - phone_present_since) >= HOLD_SECONDS:
                if (now - last_popup_time) >= COOLDOWN_SECONDS:
                    pyautogui.alert(
                        "Phone spotted 👀\nBack to work.",
                        "Get back to work"
                    )
                    last_popup_time = now
                    phone_present_since = None
        else:
            phone_present_since = None

        if not SKIP_DISPLAY:
            cv2.putText(
                frame,
                "Press Q to quit",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
            cv2.imshow("Phone Guard", frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
