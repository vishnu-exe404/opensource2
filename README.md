# # AI Exercise Rep Counter

A real-time webcam application that uses pose detection to count exercise reps (Bicep Curls, Squats, Push-ups), tracks pacing, and delivers short motivational coaching tips from a local Gemma model via Ollama — no images ever leave your machine.

## Features

- **Pose-Based Rep Counting** — Uses MediaPipe Pose to track joint angles (shoulder/elbow/wrist or hip/knee/ankle) and counts reps via an up/down state machine.
- **Multiple Exercises** — Choose between Bicep Curl, Squat, or Push-up, each with tuned angle thresholds.
- **Live Angle & Stage Overlay** — On-screen display of current joint angle, rep count, and stage ("up"/"down").
- **Local AI Coaching** — Every 5 reps (configurable), sends only the exercise name, rep count, and pace (reps/minute) — never images — to a local [Ollama](https://ollama.ai/) instance running `gemma2:2b`, and displays a short motivating tip on-screen.
- **Session Summary** — Prints total reps and elapsed time when you quit.

## Requirements

- Python 3.8+
- A webcam
- [Ollama](https://ollama.ai/) installed and running locally with the `gemma2:2b` model pulled (optional — the app runs without it, just without AI coaching tips)

### Python dependencies

```bash
pip install opencv-python mediapipe requests
```

## Project Structure

```
.
├── main.py              # (this script)
```

No local data files are created — this app doesn't store any images, video, or history; everything runs in memory for the current session.

## Usage

Run the script:

```bash
python main.py
```

You'll be prompted to choose an exercise:

```
Choose an exercise:
  1. Bicep Curl
  2. Squat
  3. Push-up
Enter choice:
```

The webcam feed opens with:
- Live pose skeleton overlay
- Rep counter and current stage (up/down)
- Current joint angle
- Periodic AI coaching tips

Press `q` or `ESC` to quit and see your session summary.

## Configuration

Key parameters can be adjusted near the top of the script:

| Variable | Description |
|---|---|
| `OLLAMA_URL` | Ollama API endpoint |
| `MODEL_NAME` | Ollama model used for coaching tips (default: `gemma2:2b`) |
| `REQUEST_COOLDOWN` | Minimum seconds between Gemma requests |
| `FEEDBACK_EVERY_N_REPS` | How often (in reps) to request new AI feedback |
| `EXERCISES` | Dict defining each exercise's tracked joints and up/down angle thresholds |

### Adding a new exercise

Add an entry to the `EXERCISES` dict with:
- `name` — display name
- `landmarks` — a 3-tuple of MediaPipe `PoseLandmark` names defining the joint angle to track (e.g. `("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE")`)
- `down_angle_max` — angle threshold for the contracted/"down" position
- `up_angle_min` — angle threshold for the extended/"up" position

## Privacy Note

No video, images, or pose data are ever saved to disk or sent externally. Only the exercise name, rep count, and pace are sent to your **local** Ollama instance to generate coaching text.

## License

Add a license of your choice (e.g. MIT) here.
