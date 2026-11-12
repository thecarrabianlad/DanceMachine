DANCE COACH - runs fully offline on your PC

1. Install Python 3.9 - 3.12, then in this folder run:
      pip install -r requirements.txt
2. Put your dance videos (.mp4/.avi/.mov) in the "videos" folder.
3. Start:
      python dance_coach.py
   or  python dance_coach.py videos\mydance.mp4

Tips
- Stand 2-3 m from the webcam so head-to-feet is visible, with good light.
- First run of a video analyses it once (progress bar); later runs start instantly.
- Camera is mirrored like a real mirror. If the dancer in the video faces
  AWAY from you, run with --no-mirror.
- Install ffmpeg (ffplay) to hear the video's music while you dance.
- Keys: Up/Down+Enter menu, Q/Esc quit, R retry, M menu.
