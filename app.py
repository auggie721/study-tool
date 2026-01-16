import time
import cv2
import pyautogui
from ultralytics import YOLO

# --- Settings ---
CONF_THRESH = 0.45          # detection confidence threshold
HOLD_SECONDS = 2        # phone must be detected continuously
COOLDOWN_SECONDS = 0       # cooldown between popups
CAMERA_INDEX = 0            # default webcam
INFERENCE_EVERY_N_FRAMES = 15

def main():
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

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

        # Run YOLO only every N frames
        if frame_count % INFERENCE_EVERY_N_FRAMES == 0:
            last_results = model(frame, verbose=False)[0]

        phone_detected = False

        # Use last known detection results
        if last_results is not None and last_results.boxes is not None:
            for box in last_results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # COCO class 67 = cell phone
                if cls_id == 67 and conf >= CONF_THRESH:
                    phone_detected = True

                    # Draw bounding box (debug)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
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
