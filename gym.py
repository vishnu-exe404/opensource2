import time
import math
import cv2
import mediapipe as mp
import requests

# --- Config ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma2:2b"
REQUEST_COOLDOWN = 8.0          # min seconds between Gemma calls
FEEDBACK_EVERY_N_REPS = 5       # ask Gemma for feedback every N reps

# Angle thresholds per exercise: (down_angle_max, up_angle_min)
# down_angle_max = angle at the "down"/contracted position
# up_angle_min = angle at the "up"/extended position
EXERCISES = {
    "1": {
        "name": "Bicep Curl",
        "landmarks": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
        "down_angle_max": 50,
        "up_angle_min": 150
    },
    "2": {
        "name": "Squat",
        "landmarks": ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
        "down_angle_max": 100,
        "up_angle_min": 160
    },
    "3": {
        "name": "Push-up",
        "landmarks": ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
        "down_angle_max": 90,
        "up_angle_min": 160
    }
}

_last_request_time = 0.0
_last_response = ""


# --- Gemma / Ollama ---
def ask_gemma(exercise_name, rep_count, elapsed_seconds):
    """
    Sends only rep count, exercise name, and pace (no images) to Gemma
    for a short coaching remark. Cooldown-limited to avoid spamming Ollama.
    """
    global _last_request_time, _last_response

    now = time.time()
    if now - _last_request_time < REQUEST_COOLDOWN:
        return _last_response

    pace = rep_count / (elapsed_seconds / 60) if elapsed_seconds > 0 else 0  # reps per minute

    prompt = (
        f"A person is doing {exercise_name}. They have completed {rep_count} reps "
        f"at a pace of about {pace:.1f} reps per minute. "
        "In one short, motivating sentence, give encouragement or a brief "
        "form/pacing tip appropriate for this exercise. "
        "Do not mention that you are an AI or reference any image or camera."
    )

    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=8)
        response.raise_for_status()
        text = response.json().get("response", "").strip()
        _last_response = text if text else "(No response from Gemma.)"
    except requests.exceptions.ConnectionError:
        _last_response = "[Ollama not reachable - is 'ollama serve' running?]"
    except requests.exceptions.Timeout:
        _last_response = "[Ollama request timed out.]"
    except Exception as e:
        _last_response = f"[Gemma error: {e}]"

    _last_request_time = now
    return _last_response


# --- Angle Calculation ---
def calculate_angle(a, b, c):
    """
    Calculates the angle (in degrees) at point b, formed by points a-b-c.
    Each point is (x, y) in pixel coordinates.
    """
    a = (a.x, a.y)
    b = (b.x, b.y)
    c = (c.x, c.y)

    ang = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) -
        math.atan2(a[1] - b[1], a[0] - b[0])
    )
    ang = abs(ang)
    if ang > 180:
        ang = 360 - ang
    return ang


def choose_exercise():
    print("Choose an exercise:")
    for key, ex in EXERCISES.items():
        print(f"  {key}. {ex['name']}")
    choice = input("Enter choice: ").strip()
    return EXERCISES.get(choice)


def draw_overlay(frame, exercise_name, rep_count, stage, angle, gemma_text):
    cv2.rectangle(frame, (0, 0), (300, 90), (245, 117, 16), -1)
    cv2.putText(frame, exercise_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Reps: {rep_count}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Stage: {stage or '-'}", (10, 82), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)

    if angle is not None:
        cv2.putText(frame, f"{angle:.0f} deg", (310, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

    if gemma_text:
        max_chars = 60
        lines = [gemma_text[i:i + max_chars] for i in range(0, len(gemma_text), max_chars)]
        for i, line in enumerate(lines[:2]):
            cv2.putText(frame, line, (10, frame.shape[0] - 40 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def run_rep_counter():
    exercise = choose_exercise()
    if exercise is None:
        print("Invalid exercise choice.")
        return

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam. Check the camera connection/permissions.")
        return

    rep_count = 0
    stage = None  # "up" or "down"
    last_feedback_rep = 0
    last_gemma_text = ""
    start_time = time.time()

    a_name, b_name, c_name = exercise["landmarks"]

    print(f"Starting {exercise['name']} rep counter. Press 'q' or ESC to quit.")

    with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to read from webcam.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_frame)

            angle = None

            if result.pose_landmarks:
                mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                lm = result.pose_landmarks.landmark
                try:
                    a = lm[getattr(mp_pose.PoseLandmark, a_name).value]
                    b = lm[getattr(mp_pose.PoseLandmark, b_name).value]
                    c = lm[getattr(mp_pose.PoseLandmark, c_name).value]
                    angle = calculate_angle(a, b, c)
                except (IndexError, AttributeError):
                    angle = None

                if angle is not None:
                    # --- Rep counting state machine ---
                    if angle >= exercise["up_angle_min"]:
                        stage = "up"
                    elif angle <= exercise["down_angle_max"] and stage == "up":
                        stage = "down"
                        rep_count += 1

                        # --- AI Coaching Feedback (only rep count/pace sent, no images) ---
                        if rep_count - last_feedback_rep >= FEEDBACK_EVERY_N_REPS:
                            elapsed = time.time() - start_time
                            last_gemma_text = ask_gemma(exercise["name"], rep_count, elapsed)
                            last_feedback_rep = rep_count

            else:
                cv2.putText(frame, "No person detected", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            draw_overlay(frame, exercise["name"], rep_count, stage, angle, last_gemma_text)

            cv2.imshow("Exercise Rep Counter", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    elapsed_total = time.time() - start_time
    print(f"\nSession complete: {rep_count} reps of {exercise['name']} "
          f"in {elapsed_total:.0f} seconds.")


if __name__ == "__main__":
    run_rep_counter()
