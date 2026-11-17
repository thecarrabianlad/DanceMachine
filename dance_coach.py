"""
Dance Coach - local desktop app (no internet needed).

Plays a reference dance video, watches you through the webcam, compares your
body pose with the dancer's pose in real time, and gives a live score.

Run:   python dance_coach.py            (shows a menu of videos in ./videos)
       python dance_coach.py my.mp4     (play a specific video)
Options: --camera 1   use another webcam
         --no-mirror  don't mirror the camera (use if the dancer faces away)
"""
import argparse
import math
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    sys.exit("MediaPipe missing. Run:  pip install -r requirements.txt")

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(HERE, "videos")
CACHE_DIR = os.path.join(HERE, ".pose_cache")
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")
PANEL_H = 480            # height of each panel (reference / camera)
WINDOW = "Dance Coach"

# Body segments scored: (name, landmark_a, landmark_b). Indices = MediaPipe Pose.
SEGMENTS = [
    ("L upper arm", 11, 13), ("L forearm", 13, 15),
    ("R upper arm", 12, 14), ("R forearm", 14, 16),
    ("L thigh", 23, 25), ("L shin", 25, 27),
    ("R thigh", 24, 26), ("R shin", 26, 28),
    ("L torso", 11, 23), ("R torso", 12, 24),
]
FULL_MARK_DEG = 15       # error below this = full points for a segment
ZERO_MARK_DEG = 60       # error above this = 0 points
SYNC_WINDOW_SEC = 0.25   # forgiveness for timing (+/- seconds)
VIS_MIN = 0.5

mp_pose = mp.solutions.pose


# ---------------------------------------------------------------- pose maths
def segment_angles(lm):
    """lm: (33,3) array x,y,visibility -> (angles[N], valid[N])."""
    ang = np.zeros(len(SEGMENTS), np.float32)
    ok = np.zeros(len(SEGMENTS), bool)
    for i, (_, a, b) in enumerate(SEGMENTS):
        if lm[a, 2] > VIS_MIN and lm[b, 2] > VIS_MIN:
            dx, dy = lm[b, 0] - lm[a, 0], lm[b, 1] - lm[a, 1]
            ang[i] = math.degrees(math.atan2(dy, dx))
            ok[i] = True
    return ang, ok


def angle_diff(a, b):
    d = np.abs(a - b) % 360
    return np.minimum(d, 360 - d)


def landmarks_to_array(results):
    if not results.pose_landmarks:
        return None
    return np.array([[p.x, p.y, p.visibility] for p in results.pose_landmarks.landmark],
                    np.float32)


def compare(user_ang, user_ok, ref_angles, ref_valid, center, win):
    """Best match of the user's pose against reference frames near `center`.
    Returns (score 0-100, per-segment error array, matched mask)."""
    lo, hi = max(0, center - win), min(len(ref_angles) - 1, center + win)
    best = None
    for j in range(lo, hi + 1):
        both = user_ok & ref_valid[j]
        if both.sum() < 4:
            continue
        err = angle_diff(user_ang, ref_angles[j])
        pts = np.clip(1 - (err - FULL_MARK_DEG) / (ZERO_MARK_DEG - FULL_MARK_DEG), 0, 1)
        score = pts[both].mean() * 100
        if best is None or score > best[0]:
            best = (score, err, both)
    return best


# ------------------------------------------------------ reference analysis
def analyse_reference(path, canvas):
    """Run pose detection on every frame of the reference video (cached)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = f"{os.path.basename(path)}_{os.path.getsize(path)}.npz"
    cache = os.path.join(CACHE_DIR, key)
    if os.path.exists(cache):
        d = np.load(cache)
        return d["angles"], d["valid"], float(d["fps"])

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    angles, valid = [], []
    with mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            lm = landmarks_to_array(pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            if lm is None:
                angles.append(np.zeros(len(SEGMENTS), np.float32))
                valid.append(np.zeros(len(SEGMENTS), bool))
            else:
                a, v = segment_angles(lm)
                angles.append(a)
                valid.append(v)
            n += 1
            if n % 5 == 0:
                img = backdrop(300, 800)
                text(img, "Analysing dance video (one time only)...", (40, 120), 0.9)
                bar = int(720 * n / total)
                cv2.rectangle(img, (40, 170), (760, 200), (80, 80, 80), 2)
                cv2.rectangle(img, (40, 170), (40 + bar, 200), (0, 220, 120), -1)
                cv2.imshow(WINDOW, img)
                if cv2.waitKey(1) & 0xFF == 27:
                    sys.exit(0)
    cap.release()
    angles, valid = np.array(angles), np.array(valid)
    np.savez(cache, angles=angles, valid=valid, fps=fps)
    return angles, valid, fps


# --------------------------------------------------------------- drawing
# OpenCV uses BGR colors. Keep the visual theme shared across screens.
CYAN = (240, 215, 0)
PINK = (155, 45, 245)
GOLD = (30, 215, 255)
MUTED = (160, 175, 185)


def backdrop(h, w):
    img = np.full((h, w, 3), (17, 12, 9), np.uint8)
    for y in range(0, h, 32):
        cv2.line(img, (0, y), (w, y), (27, 23, 16), 1)
    for x in range(0, w, 64):
        cv2.line(img, (x, 0), (x, h), (24, 20, 14), 1)
    return img


def frame(img, x, y, w, h, color):
    cut = min(14, w // 5, h // 5)
    points = np.array([(x + cut, y), (x + w - cut, y),
                       (x + w, y + cut), (x + w, y + h - cut),
                       (x + w - cut, y + h), (x + cut, y + h),
                       (x, y + h - cut), (x, y + cut)], np.int32)
    cv2.polylines(img, [points], True, tuple(c // 4 for c in color), 6, cv2.LINE_AA)
    cv2.polylines(img, [points], True, color, 1, cv2.LINE_AA)
    cv2.line(img, (x + cut, y), (x + min(w // 3, 90), y), color, 3, cv2.LINE_AA)


def dashboard(ref_img, cam_img, avg, label, combo, shown_score, tracking, elapsed, duration):
    # Equal panels letterbox each source so portrait videos retain their full pose.
    panel_w, panel_h = 640, PANEL_H
    img = backdrop(744, 1336)
    cards = [(20, 230, CYAN), (266, 338, CYAN),
             (620, 370, GOLD), (1006, 310, PINK)]
    for x, w, color in cards:
        frame(img, x, 20, w, 100, color)
    text(img, "DANCE", (42, 62), 1.1, CYAN, 2)
    text(img, "COACH", (42, 101), 1.1, CYAN, 2)
    text(img, "SCORE", (286, 47), 0.55, CYAN)
    text(img, f"{avg:04.1f}", (286, 99), 1.65, CYAN, 3)
    text(img, "/ 100", (464, 96), 0.85, MUTED)
    text(img, label[0] or "GET READY", (646, 84), 1.25, GOLD, 3)
    text(img, "COMBO", (1028, 47), 0.55, PINK)
    text(img, str(combo), (1028, 100), 1.65, PINK, 3)
    cv2.rectangle(img, (20, 136), (1316, 148), (49, 40, 31), -1)
    fill = int(1296 * np.clip(shown_score / 100, 0, 1))
    if fill:
        cv2.rectangle(img, (20, 136), (20 + fill, 148), CYAN, -1)
    status = "POSE TRACKING" if tracking else "POSE NOT DETECTED"
    text(img, "CAMERA CONNECTED", (200, 180), 0.55, CYAN, 1)
    text(img, status, (550, 180), 0.55, CYAN if tracking else PINK, 1)
    text(img, "LIVE SESSION", (940, 180), 0.55, MUTED, 1)
    for source, x, title, color in ((ref_img, 20, "DANCER / REFERENCE", CYAN),
                                    (cam_img, 676, "YOU / LIVE CAMERA", PINK)):
        text(img, title, (x + 14, 216), 0.65, color)
        scale = min(panel_w / source.shape[1], panel_h / source.shape[0])
        fitted = cv2.resize(source, (max(1, int(source.shape[1] * scale)),
                                     max(1, int(source.shape[0] * scale))))
        y = 230 + (panel_h - fitted.shape[0]) // 2
        left = x + (panel_w - fitted.shape[1]) // 2
        img[y:y + fitted.shape[0], left:left + fitted.shape[1]] = fitted
        frame(img, x, 230, panel_w, panel_h, color)
    def timestamp(seconds):
        seconds = max(0, int(seconds))
        return f"{seconds // 60:02d}:{seconds % 60:02d}"
    text(img, f"{timestamp(elapsed)} / {timestamp(duration)}", (34, 734), 0.5, MUTED, 1)
    text(img, "Q / ESC  EXIT", (1134, 734), 0.5, MUTED, 1)
    return img


def text(img, s, org, scale=0.7, color=(255, 255, 255), thick=2):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def fit_height(img, h):
    s = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * s), h))


def draw_user_skeleton(img, lm, err, matched):
    """Green = segment matches the dancer, red = off."""
    h, w = img.shape[:2]
    pt = lambda i: (int(lm[i, 0] * w), int(lm[i, 1] * h))
    for k, (_, a, b) in enumerate(SEGMENTS):
        if lm[a, 2] < VIS_MIN or lm[b, 2] < VIS_MIN:
            continue
        col = (200, 200, 200)
        if err is not None and matched[k]:
            col = (150, 255, 40) if err[k] <= 30 else PINK
        cv2.line(img, pt(a), pt(b), col, 3, cv2.LINE_AA)
        cv2.circle(img, pt(a), 6, (255, 255, 255), -1)
        cv2.circle(img, pt(b), 6, (255, 255, 255), -1)


def judgement(s):
    if s >= 85: return "PERFECT!", (0, 255, 255)
    if s >= 70: return "GREAT", (0, 230, 0)
    if s >= 50: return "GOOD", (255, 200, 0)
    return "MISS", (60, 60, 255)


def grade(avg):
    for g, t in (("S", 85), ("A", 72), ("B", 58), ("C", 42)):
        if avg >= t:
            return g
    return "D"


# ------------------------------------------------------------------ audio
class Audio:
    """Optional: plays the video's sound through ffplay if it is installed."""
    def __init__(self, path):
        self.proc = None
        exe = shutil.which("ffplay")
        if exe:
            self.proc = subprocess.Popen([exe, "-nodisp", "-autoexit", "-loglevel", "quiet", path],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


# ------------------------------------------------------------------ screens
def menu(videos):
    sel = 0
    while True:
        img = backdrop(520, 900)
        text(img, "DANCE COACH", (40, 70), 1.4, CYAN, 3)
        text(img, "Up/Down + Enter to choose   |   Q to quit", (40, 110), 0.6, (180, 180, 180), 1)
        if not videos:
            text(img, f"Put dance videos (.mp4) in: {VIDEO_DIR}", (40, 220), 0.7, (80, 80, 255))
        for i, v in enumerate(videos[:10]):
            col = CYAN if i == sel else MUTED
            text(img, ("> " if i == sel else "  ") + os.path.basename(v), (60, 170 + i * 34), 0.8, col)
        cv2.imshow(WINDOW, img)
        k = cv2.waitKeyEx(50)
        if k in (ord("q"), 27):
            return None
        if k in (2490368, 82, ord("w")): sel = max(0, sel - 1)            # up
        if k in (2621440, 84, ord("s")): sel = min(len(videos) - 1, sel + 1)  # down
        if k in (13, 10) and videos:
            return videos[sel]


def countdown(cam, mirror, seconds=3):
    end = time.time() + seconds
    while time.time() < end:
        ok, f = cam.read()
        if not ok:
            return
        if mirror: f = cv2.flip(f, 1)
        f = fit_height(f, PANEL_H)
        n = int(end - time.time()) + 1
        text(f, str(n), (f.shape[1] // 2 - 40, PANEL_H // 2 + 40), 4, (0, 255, 255), 8)
        text(f, "Step back so your whole body is visible!", (20, 40), 0.7, (255, 255, 255))
        cv2.imshow(WINDOW, f)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            return


def play(path, cam, mirror):
    angles, valid, fps = analyse_reference(path, None)
    ref = cv2.VideoCapture(path)
    win = max(1, int(SYNC_WINDOW_SEC * fps))
    countdown(cam, mirror)

    audio = Audio(path)
    start = time.time()
    ref_pos, ref_frame = -1, None
    scores, recent = [], []
    combo = best_combo = 0
    shown_score, label = 0.0, ("", (255, 255, 255))
    err, matched = None, None

    with mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while True:
            t = time.time() - start
            target = int(t * fps)
            # advance reference video to the current time
            while ref_pos < target:
                ok, fr = ref.read()
                if not ok:
                    audio.stop()
                    return scores, best_combo
                ref_pos, ref_frame = ref_pos + 1, fr

            ok, cf = cam.read()
            if not ok:
                break
            if mirror:
                cf = cv2.flip(cf, 1)
            lm = landmarks_to_array(pose.process(cv2.cvtColor(cf, cv2.COLOR_BGR2RGB)))
            cam_img = fit_height(cf, PANEL_H)

            if lm is not None and ref_pos < len(angles) and valid[ref_pos].sum() >= 4:
                ua, uok = segment_angles(lm)
                res = compare(ua, uok, angles, valid, ref_pos, win)
                if res:
                    raw, err, matched = res
                    recent = (recent + [raw])[-6:]
                    smooth = sum(recent) / len(recent)
                    scores.append(smooth)
                    shown_score += (smooth - shown_score) * 0.2
                    label = judgement(smooth)
                    combo = combo + 1 if smooth >= 70 else 0
                    best_combo = max(best_combo, combo)
                draw_user_skeleton(cam_img, lm, err, matched) if err is not None else None
            elif lm is None:
                text(cam_img, "No body detected - step back", (20, 80), 0.7, (80, 80, 255))

            ref_img = fit_height(ref_frame, PANEL_H)
            avg = sum(scores) / len(scores) if scores else 0
            canvas = dashboard(ref_img, cam_img, avg, label, combo,
                               shown_score, lm is not None, t, len(angles) / fps)
            cv2.imshow(WINDOW, canvas)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                audio.stop()
                return scores, best_combo


def result_screen(scores, best_combo):
    avg = sum(scores) / len(scores) if scores else 0
    g = grade(avg)
    while True:
        img = backdrop(520, 900)
        text(img, "RESULTS", (40, 80), 1.4, CYAN, 3)
        text(img, f"Grade: {g}", (40, 190), 2.6, PINK, 5)
        text(img, f"Score: {avg:.1f} / 100", (40, 270), 1.0)
        text(img, f"Best combo: {best_combo}", (40, 320), 1.0)
        text(img, "R = retry     M = menu     Q = quit", (40, 450), 0.8, (180, 180, 180), 1)
        cv2.imshow(WINDOW, img)
        k = cv2.waitKey(50) & 0xFF
        if k == ord("r"): return "retry"
        if k == ord("m"): return "menu"
        if k in (ord("q"), 27): return "quit"


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-mirror", action="store_true")
    args = ap.parse_args()
    mirror = not args.no_mirror

    cam = cv2.VideoCapture(args.camera)
    if not cam.isOpened():
        sys.exit("Cannot open the webcam. Try --camera 1")
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    os.makedirs(VIDEO_DIR, exist_ok=True)
    path = args.video
    while True:
        if not path:
            vids = sorted(os.path.join(VIDEO_DIR, f) for f in os.listdir(VIDEO_DIR)
                          if f.lower().endswith(VIDEO_EXT))
            path = menu(vids)
            if not path:
                break
        scores, combo = play(path, cam, mirror)
        action = result_screen(scores, combo)
        if action == "quit":
            break
        if action == "menu":
            path = None
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
