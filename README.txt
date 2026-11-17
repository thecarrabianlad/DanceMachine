DANCE COACH - runs fully offline on your PC

1. Install Python 3.9 - 3.12, then in this folder run:
      pip install -r requirements.txt
2. Put your dance videos (.mp4/.avi/.mov) in the "videos" folder.
3. Start:
      python dance_coach.py
   or  python dance_coach.py videos\mydance.mp4

Tips
- Stand 2-3 m from the webcam so head-to-feet is visible, with good light.
- Gameplay runs in 2 separate windows (Dancer video & Your Camera) so you can resize or move them to separate screens/TVs.
- First run of a video analyses it once (progress bar); later runs start instantly.
- 3D Rhythm Scoring:
  * Uses 3D world landmarks (perspective and tilt invariant).
  * Dynamic Motion Weighting: Limbs actively moving in the choreography count up to 5x more than stationary limbs.
  * Rhythm Timing: Awards bonus points for hitting poses on beat, with feedback like "PERFECT!", "GREAT (LATE)", "GREAT (EARLY)", etc.
  * Skeleton Colors: Green (accurate), Yellow (close), Red (miss). Glowing points indicate active moving limbs.
- Camera is mirrored like a real mirror. If the dancer in the video faces
  AWAY from you, run with --no-mirror.
- Install ffmpeg (ffplay) to hear the video's music while you dance.
- Keys: Up/Down+Enter menu, Q/Esc quit, R retry, M menu.
