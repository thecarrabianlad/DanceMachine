DANCE COACH - runs fully offline on your PC

1. Install Python 3.9 - 3.12, then in this folder run:
      pip install -r requirements.txt
2. Put your dance videos (.mp4/.avi/.mov) in the "videos" folder.
3. Start:
      python dance_coach.py
   or  python dance_coach.py videos\mydance.mp4

Tips
- Stand 2-3 m from the webcam so head-to-feet is visible, with good light.
- Gameplay shows the dancer and webcam side by side in fullscreen.
- First run of a video analyses it once (progress bar); later runs start instantly.
- Scoring:
  * Uses 3D world landmarks and analyzes every reference frame on the first run.
  * Moving limbs get more weight. Sustained whole-body stillness scores zero during reference movement; stepping and body translation count as movement.
  * Frames where pose tracking fails count as misses instead of being omitted from the session average.
  * The final score is hit rate: every evaluated pose at 50/100 similarity or higher earns one point; lower matches, stillness, and tracking failures earn zero.
  * The app matches a short phrase to auto-align up to six seconds of start delay, then tracks gradual tempo drift.
  * Feedback shows pose and timing judgements such as "PERFECT!", "LATE", and "OFF BEAT".
  * Skeleton Colors: Green (accurate), Yellow (close), Red (miss). Glowing points indicate active moving limbs.
- Camera is mirrored like a real mirror. If the dancer in the video faces
  AWAY from you, run with --no-mirror.
- Install ffmpeg (ffplay) to hear the video's music while you dance.
- Keys: Up/Down+Enter menu, Q/Esc quit, R retry, M menu.
