<p align="center">
  <img src="assets/dance-machine-logo.png" alt="Dance Machine" width="420">
</p>

<h1 align="center">Dance Machine</h1>

<p align="center"><strong>Turn a dance video into an offline, webcam-powered practice session.</strong></p>

Dance Machine plays a dance video beside your webcam feed, tracks your pose, and scores how closely you follow the dancer. It runs locally on your computer; after setup, videos and webcam frames stay on your device.

## What it does

- Plays a reference video and your mirrored webcam feed side by side.
- Tracks body pose and gives timing and movement feedback as you dance.
- Aligns your movement with the video, even if you start a few seconds late.
- Shows a final hit rate and best combo when the session ends.
- Analyzes each reference video once, then stores the pose data in `.pose_cache` for faster later runs.
- Works offline after its Python dependencies are installed. `ffplay` is optional and only needed for video sound.

## Device support

This is a **desktop Python application**, not a mobile app or a browser app. It needs a desktop display, a webcam that OpenCV can access, and a keyboard or mouse. Use a 64-bit desktop environment with CPython 3.9–3.12. The pinned MediaPipe package provides wheels for Windows x86-64, macOS (Intel and Apple Silicon), and Linux x86-64/ARM64; see the [MediaPipe 0.10.14 wheel list](https://pypi.org/project/mediapipe/0.10.14/#files) for details. The pinned package does not list a Windows ARM64 wheel.

Python 3.11 is a good starting point. The first dependency install needs an internet connection. Gameplay itself is local.

## Quick start

1. Install Git and Python 3.9–3.12 (64-bit). Python 3.11 is recommended.
2. Clone this repository and enter its folder:

   ```sh
   git clone https://github.com/thecarrabianlad/DanceMachine.git
   cd DanceMachine
   ```

   Or download the repository as a ZIP from GitHub and open the extracted `DanceMachine` folder in a terminal.

3. Create a virtual environment, install the dependencies, and launch the app using the instructions for your operating system below.
4. Allow camera access when your operating system asks.
5. Choose one of the sample videos, or add your own video to the `videos` folder.

## Install and run

### Windows 10 or 11

Open PowerShell in the project folder. Install Python 3.11 from [python.org](https://www.python.org/downloads/) with the Python Launcher enabled, then run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe dance_coach.py
```

If Windows blocks camera access, open **Settings → Privacy & security → Camera** and allow desktop apps to use the camera.

### macOS

Open Terminal in the project folder. Install Python 3.11 (for example, from [python.org](https://www.python.org/downloads/)), then run:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python dance_coach.py
```

If macOS blocks the webcam, open **System Settings → Privacy & Security → Camera** and allow access for the terminal or Python app you use to launch Dance Machine.

### Linux

On Ubuntu or Debian, install Python, Tk, and the OpenGL runtime libraries first:

```sh
sudo apt update
sudo apt install python3 python3-venv python3-tk libgl1
```

Make sure `python3 --version` reports Python 3.9–3.12. If it does not, install a supported Python version for your distribution. Then, from the project folder, run:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python dance_coach.py
```

Other Linux distributions need the equivalent Python venv, Tk, and OpenGL packages. If your desktop is Wayland and OpenCV reports that its Qt Wayland plugin is missing, try launching from an XWayland session with:

```sh
QT_QPA_PLATFORM=xcb python dance_coach.py
```

## Add dance videos

Copy videos into the repository's `videos` folder. The menu recognizes `.mp4`, `.avi`, `.mov`, `.mkv`, and `.webm` files. Restart Dance Machine after adding files.

You can also start directly with a video path:

```sh
python dance_coach.py "videos/my dance.mp4"
```

On Windows, use the Python executable inside the virtual environment:

```powershell
.\.venv\Scripts\python.exe dance_coach.py "videos\my dance.mp4"
```

## Controls

| Screen | Controls |
| --- | --- |
| Video menu | Arrow keys or W/A/S/D to select; Enter to play; click a video tile to start it; Q or Esc to quit |
| Dance session | Q or Esc to end the session and view results |
| Results | R to retry; M to return to the menu; Q or Esc to quit |

The webcam view is mirrored by default, like a mirror. Use `--no-mirror` if the dancer in the reference video faces away from you. If the wrong webcam opens, select another camera index with `--camera`:

```sh
python dance_coach.py --camera 1
python dance_coach.py "videos/my dance.mp4" --camera 1 --no-mirror
```

## Tips for a good session

- Place the camera far enough away to keep your whole body, from head to feet, in view. About 2–3 m is a useful starting point.
- Use a well-lit space and avoid standing in front of a bright window.
- Keep the reference dancer's full body visible in the video when possible.
- On the first run of a video, wait for its pose-analysis progress to finish. Later runs reuse the saved analysis in `.pose_cache`.
- Every evaluated pose at 50/100 similarity or higher counts as a hit. Tracking misses and sustained stillness while the reference dancer is moving count as misses.

## Optional video audio

Dance Machine can play the reference video's sound through `ffplay` when it is installed. Install FFmpeg from the [official FFmpeg download page](https://ffmpeg.org/download.html), making sure `ffplay` is on your `PATH`. The app still works without it.

## Troubleshooting

**“Cannot open the webcam.”** Allow camera access in your OS settings, close other apps using the camera, then try another camera index with `--camera 1`.

**The webcam view is black or no pose is detected.** Improve the lighting, move back until your full body is visible, and check that the correct camera is selected.

**`pip install` cannot find a MediaPipe wheel.** Confirm that you are using 64-bit CPython 3.9–3.12 on a supported desktop platform. Check the [MediaPipe 0.10.14 wheel list](https://pypi.org/project/mediapipe/0.10.14/#files) for your Python version and operating system.

**OpenCV reports a missing Qt or display plugin.** Make sure you are running from a desktop session with a display. On Wayland, try the XWayland command shown in the Linux instructions.

## Dependencies

The runtime dependencies are listed in [`requirements.txt`](requirements.txt): OpenCV, NumPy, MediaPipe, and Pillow. Dance Machine does not require a cloud account or send webcam footage to a service.
