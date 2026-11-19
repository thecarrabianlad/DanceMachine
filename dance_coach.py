"""
Dance Coach - local desktop app (no internet needed).

Plays a reference dance video, watches you through the webcam, compares your
body pose with the dancer's pose in real time, and gives a live score.

Run:   python dance_coach.py            (shows a menu of videos in ./videos)
       python dance_coach.py my.mp4     (play a specific video)
Options: --camera 1   use another webcam
         --no-mirror  don't mirror the camera (use if the dancer faces away)
"""
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageSequence
from functools import lru_cache
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
WIN_VIDEO = "Dance Coach - Video"
WIN_CAM = "Dance Coach - Camera"

# Body segments scored: (name, landmark_a, landmark_b). Indices = MediaPipe Pose.
SEGMENTS = [
    ("L upper arm", 11, 13), ("L forearm", 13, 15),
    ("R upper arm", 12, 14), ("R forearm", 14, 16),
    ("L thigh", 23, 25), ("L shin", 25, 27),
    ("R thigh", 24, 26), ("R shin", 26, 28),
    ("L torso", 11, 23), ("R torso", 12, 24),
]
FULL_MARK_DEG = 8        # error below this = full points for a segment
ZERO_MARK_DEG = 45       # error above this = 0 points
SYNC_WINDOW_SEC = 0.50   # follow gradual tempo drift while preserving phase continuity
VIS_MIN = 0.35
STILL_WINDOW = 8         # compare poses across several camera frames to reject jitter
STILL_BODY_MOTION = 0.035  # body-scale displacement threshold over STILL_WINDOW
HIT_THRESHOLD = 50       # minimum similarity that earns a move hit and combo credit
MAX_LAG_SEC = 6.0        # auto-align users who start late or dance at a different tempo
ALIGN_HISTORY = 18       # require a short movement phrase before locking alignment
ALIGN_RECHECK = 6
ALIGN_MOTION_DEG = 2.0   # allow small, slower movements to establish alignment

mp_pose = mp.solutions.pose


# ---------------------------------------------------------------- pose maths
def extract_landmarks(results):
    """Extracts (screen_lm, world_lm) from MediaPipe Pose results.
    screen_lm: (33, 3) array [x, y, visibility] in 2D image coordinates.
    world_lm:  (33, 4) array [x, y, z, visibility] in metric 3D space.
    """
    if not results or not results.pose_landmarks:
        return None, None
    screen_lm = np.array([[p.x, p.y, p.visibility] for p in results.pose_landmarks.landmark], np.float32)
    world_lm = None
    if results.pose_world_landmarks:
        world_lm = np.array([[p.x, p.y, p.z, p.visibility] for p in results.pose_world_landmarks.landmark], np.float32)
    return screen_lm, world_lm


def segment_vectors(world_lm, screen_lm=None, vis_min=VIS_MIN):
    """Computes 3D unit bone direction vectors for all SEGMENTS.
    Falls back to 2D screen vectors if world_lm is temporarily unavailable.
    Returns (vecs[10, 3], valid[10]).
    """
    vecs = np.zeros((len(SEGMENTS), 3), np.float32)
    ok = np.zeros(len(SEGMENTS), bool)
    if world_lm is not None:
        for i, (_, a, b) in enumerate(SEGMENTS):
            if world_lm[a, 3] > vis_min and world_lm[b, 3] > vis_min:
                d = world_lm[b, :3] - world_lm[a, :3]
                norm = np.linalg.norm(d)
                if norm > 1e-4:
                    vecs[i] = d / norm
                    ok[i] = True
    elif screen_lm is not None:
        for i, (_, a, b) in enumerate(SEGMENTS):
            if screen_lm[a, 2] > vis_min and screen_lm[b, 2] > vis_min:
                d = np.array([screen_lm[b, 0] - screen_lm[a, 0],
                              screen_lm[b, 1] - screen_lm[a, 1], 0.0], np.float32)
                norm = np.linalg.norm(d)
                if norm > 1e-4:
                    vecs[i] = d / norm
                    ok[i] = True
    return vecs, ok


def body_motion(current, previous):
    """Return robust landmark displacement, normalized by torso length.
    Retains whole-body translation, so stepping in place counts as movement.
    """
    if current is None or previous is None:
        return None
    tracked = np.array([11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28])
    common = (current[tracked, 2] >= VIS_MIN) & (previous[tracked, 2] >= VIS_MIN)
    if common.sum() < 4:
        return None
    torso_now = (np.linalg.norm(current[11, :2] - current[23, :2]) +
                 np.linalg.norm(current[12, :2] - current[24, :2])) * 0.5
    torso_old = (np.linalg.norm(previous[11, :2] - previous[23, :2]) +
                 np.linalg.norm(previous[12, :2] - previous[24, :2])) * 0.5
    scale = max(1e-4, (torso_now + torso_old) * 0.5)
    shift = np.linalg.norm(current[tracked[common], :2] - previous[tracked[common], :2], axis=1) / scale
    return float(np.percentile(shift, 75))


def pose_similarity(user_vecs, user_ok, ref_vecs, ref_valid, ref_weights):
    """Score one pose without timing, for sequence-based phase alignment."""
    both = user_ok & ref_valid
    if both.sum() < 4:
        return None
    dots = np.clip((user_vecs[both] * ref_vecs[both]).sum(axis=-1), -1.0, 1.0)
    err = np.degrees(np.arccos(dots))
    points = np.clip(1.0 - (np.maximum(0.0, err - 8.0) / 37.0) ** 1.6, 0.0, 1.0)
    weights = ref_weights[both]
    return float((points * weights).sum() / max(1e-6, weights.sum()) * 100.0)


def estimate_lag(history, ref_vecs, ref_valid, ref_weights, fps):
    """Find a stable time offset by matching a phrase, not one lucky pose."""
    changes = []
    for (_, prev_vecs, prev_ok), (_, curr_vecs, curr_ok) in zip(history, history[1:]):
        both = prev_ok & curr_ok
        if both.any():
            dots = np.clip((prev_vecs[both] * curr_vecs[both]).sum(axis=-1), -1.0, 1.0)
            changes.extend(np.degrees(np.arccos(dots)).tolist())
    if not changes or float(np.percentile(changes, 75)) < ALIGN_MOTION_DEG:
        return None
    max_lag = int(MAX_LAG_SEC * fps)
    step = max(1, int(round(fps * 0.08)))
    best_lag, best_score = None, -1.0
    for lag in range(-max_lag, max_lag + 1, step):
        vals = []
        for ref_pos, user_vecs, user_ok in history:
            j = ref_pos - lag
            if 0 <= j < len(ref_vecs):
                score = pose_similarity(user_vecs, user_ok, ref_vecs[j], ref_valid[j], ref_weights[j])
                if score is not None:
                    vals.append(score)
        coverage = len(vals) / len(history)
        if coverage < 0.7:
            continue
        score = float(np.mean(vals)) * coverage
        if score > best_score:
            best_lag, best_score = lag, score
    if best_score < 35.0:
        return None
    return float(best_lag)


def compare(user_vecs, user_ok, ref_vecs, ref_valid, ref_weights, center, win, fps):
    """Matches user's 3D pose vectors against reference frames in [center - win, center + win].
    Evaluates 3D posture, active limb weighting, and rhythm timing.
    Returns (final_score, err_array, matched_mask, timing_tag, timing_diff)
    """
    lo = max(0, center - win)
    hi = min(len(ref_vecs) - 1, center + win)
    best = None

    for j in range(lo, hi + 1):
        both = user_ok & ref_valid[j]
        if both.sum() < 4:
            continue
        # 3D angular error in degrees for each segment
        dots = np.clip((user_vecs[both] * ref_vecs[j, both]).sum(axis=-1), -1.0, 1.0)
        err = np.zeros(len(SEGMENTS), np.float32)
        err[both] = np.degrees(np.arccos(dots))

        # Steep score curve — must be precise to score well
        pts = np.clip(1.0 - ((np.maximum(0.0, err[both] - 8.0) / 37.0) ** 1.6), 0.0, 1.0)
        w = ref_weights[j, both]
        pose_score = (pts * w).sum() / w.sum() * 100.0

        # Rhythm Timing Sync — tight windows, harsh penalties
        delta_t = (j - center) / fps
        abs_dt = abs(delta_t)
        if abs_dt <= 0.04:
            timing_mult = 1.0
            tag = "ON BEAT"
        elif abs_dt <= 0.25:
            timing_mult = 0.98
            tag = "LATE" if delta_t < 0 else "EARLY"
        elif abs_dt <= 0.50:
            timing_mult = 0.95
            tag = "TOO LATE" if delta_t < 0 else "TOO EARLY"
        else:
            timing_mult = 0.90
            tag = "OFF BEAT"

        final_score = pose_score * timing_mult

        if best is None or final_score > best[0]:
            best = (final_score, err, both, tag, delta_t)

    return best


# ------------------------------------------------------ reference analysis
def analyse_reference(path, canvas):
    """Run 3D pose detection and motion velocity analysis on the reference video (cached)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = f"{os.path.basename(path)}_{os.path.getsize(path)}_rhythm_v2.npz"
    cache = os.path.join(CACHE_DIR, key)
    if os.path.exists(cache):
        d = np.load(cache)
        return d["vecs"], d["valid"], d["weights"], float(d["fps"])

    fullscreen()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    vecs, valid = [], []

    with mp_pose.Pose(model_complexity=1, min_detection_confidence=0.4,
                      min_tracking_confidence=0.4) as pose:
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            screen_lm, world_lm = extract_landmarks(res)
            v, v_ok = segment_vectors(world_lm, screen_lm)
            vecs.append(v)
            valid.append(v_ok)
            n += 1
            if n % 5 == 0:
                img = np.zeros((300, 800, 3), np.uint8)
                text(img, "Analysing 3D dance rhythm (one time only)...", (40, 120), 0.85)
                bar = int(720 * n / total)
                cv2.rectangle(img, (40, 170), (760, 200), (80, 80, 80), 2)
                cv2.rectangle(img, (40, 170), (40 + bar, 200), (0, 220, 120), -1)
                pct = int(100 * n / total)
                text(img, f"{pct}%", (700, 155), 0.65, (200, 200, 200), 1)
                show_fullscreen(img)
                if (cv2.waitKey(1) & 0xFF == 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    sys.exit(0)

    cap.release()
    vecs = np.array(vecs, np.float32)
    valid = np.array(valid, bool)
    T = len(vecs)
    weights = np.ones((T, len(SEGMENTS)), np.float32)

    # Compute dynamic motion activity weights based on segment angular velocities
    if T > 1:
        vel = np.zeros((T, len(SEGMENTS)), np.float32)
        for t in range(1, T):
            both = valid[t] & valid[t - 1]
            if both.any():
                dots = np.clip((vecs[t, both] * vecs[t - 1, both]).sum(axis=-1), -1.0, 1.0)
                vel[t, both] = np.degrees(np.arccos(dots)) * fps
        kernel = np.ones(5, np.float32) / 5.0
        smoothed_vel = np.zeros_like(vel)
        for k in range(len(SEGMENTS)):
            smoothed_vel[:, k] = np.convolve(vel[:, k], kernel, mode="same")
        weights = 1.0 + np.clip(smoothed_vel / 50.0, 0.0, 4.0)

    np.savez(cache, vecs=vecs, valid=valid, weights=weights, fps=fps)
    return vecs, valid, weights, fps


# --------------------------------------------------------------- drawing
def text(img, s, org, scale=0.7, color=(255, 255, 255), thick=2):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def fit_height(img, h):
    s = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * s), h))


def draw_user_skeleton(img, lm, err, matched, weights=None):
    """Green = accurate, yellow = close, red = off.
    Actively moving limbs are highlighted with thicker lines and glowing joints."""
    h, w = img.shape[:2]
    pt = lambda i: (int(lm[i, 0] * w), int(lm[i, 1] * h))
    for k, (_, a, b) in enumerate(SEGMENTS):
        if lm[a, 2] < VIS_MIN or lm[b, 2] < VIS_MIN:
            continue
        col = (200, 200, 200)
        thick = 4
        if err is not None and matched[k]:
            if err[k] <= 12:
                col = (0, 240, 0)      # Green: Nailed it
            elif err[k] <= 30:
                col = (0, 220, 255)    # Yellow: Close
            else:
                col = (50, 50, 255)    # Red: Off

        if weights is not None and weights[k] > 2.0:
            thick = 6
            cv2.circle(img, pt(a), 7, (0, 255, 255), -1)
            cv2.circle(img, pt(b), 7, (0, 255, 255), -1)

        cv2.line(img, pt(a), pt(b), col, thick, cv2.LINE_AA)
        cv2.circle(img, pt(a), 5, (255, 255, 255), -1)
        cv2.circle(img, pt(b), 5, (255, 255, 255), -1)


def judgement(score, timing_tag="ON BEAT"):
    if score >= 90:
        if timing_tag == "ON BEAT":
            return "PERFECT!", (0, 255, 255)
        else:
            return f"GREAT ({timing_tag})", (0, 240, 140)
    if score >= 75:
        if timing_tag == "ON BEAT":
            return "GREAT!", (0, 230, 0)
        else:
            return f"GOOD ({timing_tag})", (255, 210, 0)
    if score >= HIT_THRESHOLD:
        return "HIT", (255, 200, 0)
    if score >= 35:
        return "BAD", (80, 120, 255)
    return "MISS", (60, 60, 255)


def grade(avg):
    for g, t in (("S", 92), ("A", 80), ("B", 65), ("C", 50), ("D", 35), ("E", 20)):
        if avg >= t:
            return g
    return "F"


def draw_hud(img, title, avg, label, combo, shown_score):
    h, w = img.shape[:2]
    # Header background bar
    cv2.rectangle(img, (0, 0), (w, 54), INK, -1)
    # Average score on left
    text(img, f"HIT RATE {avg:5.1f}%", (15, 36), 0.85, GOLD, 2)
    # Judgement label centered
    if label and label[0]:
        tw = cv2.getTextSize(label[0], cv2.FONT_HERSHEY_SIMPLEX, 0.95, 2)[0][0]
        text(img, label[0], (max(10, (w - tw) // 2), 36), 0.95, label[1], 2)
    # Combo on right
    combo_str = f"COMBO {combo}"
    cw = cv2.getTextSize(combo_str, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0][0]
    text(img, combo_str, (max(w - cw - 15, w // 2 + 40), 36), 0.85, (255, 255, 255), 2)
    # Score meter bar
    cv2.rectangle(img, (0, 54), (w, 60), (45, 45, 45), -1)
    meter_w = int(w * min(1.0, max(0.0, shown_score / 100.0)))
    if meter_w > 0:
        cv2.rectangle(img, (0, 54), (meter_w, 60), LIME, -1)
    # Title badge bottom left
    text(img, title, (15, h - 14), 0.65, (200, 200, 200), 1)


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
CYAN = (235, 222, 35)
PINK = (185, 53, 255)
WHITE = (225, 246, 255)
INK = (43, 21, 38)
LIME = (51, 241, 177)
GOLD = (38, 213, 255)
ORANGE = (65, 114, 255)
BUTTON_COLORS = {"previous": CYAN, "next": PINK, "play": LIME,
                 "retry": GOLD, "menu": CYAN, "quit": ORANGE}


def panel(title):
    img = np.full((540, 960, 3), INK, np.uint8)
    for offset, color in [(0, PINK), (32, GOLD), (64, LIME)]:
        x = 808 + offset
        cv2.fillConvexPoly(img, np.int32([(x, 0), (x+23, 0), (x-10, 58), (x-33, 58)]), color)
    text(img, "DANCE COACH", (36, 36), .55, CYAN)
    text(img, title, (32, 98), 1.65, GOLD, 3)
    cv2.line(img, (35, 113), (923, 113), PINK, 2)
    return img


@lru_cache(maxsize=24)
def button_art(label, width, height):
    font_path = os.path.join(HERE, "assets", "fonts", "Drobix.ttf")
    mask = Image.new("L", (width, height))
    draw = ImageDraw.Draw(mask)
    for size in range(40, 9, -1):
        font = ImageFont.truetype(font_path, size)
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        if right-left <= width-18 and bottom-top <= height-10:
            break
    pos = ((width-right+left)//2-left, (height-bottom+top)//2-top)
    draw.text(pos, label, font=font, fill=255)
    alpha = np.asarray(mask)[:, :, None]/255.
    return alpha


def control(img, box, label, active=False):
    x, y, w, h = box
    paint = BUTTON_COLORS[label.split()[0].lower()]
    cv2.rectangle(img, (x,y), (x+w,y+h), paint, -1)
    cv2.rectangle(img, (x,y), (x+w,y+h), WHITE if active else INK, 2)
    alpha = button_art(label, w, h)
    region = img[y:y+h,x:x+w]
    shadow = np.roll(alpha, (2,2), axis=(0,1))
    region[:] = region*(1-shadow)+np.array(WHITE)*shadow
    region[:] = region*(1-alpha)+np.array(INK)*alpha


def fullscreen():
    global display_size
    from tkinter import Tk
    probe = Tk()
    probe.withdraw()
    display_size = (probe.winfo_screenwidth(), probe.winfo_screenheight())
    probe.destroy()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(WINDOW, *display_size)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def show_fullscreen(frame):
    cv2.imshow(WINDOW, cv2.resize(frame, display_size))
    return display_size


def screen_input(render, animation=None):
    fullscreen()
    hovered = pressed = None
    pointer = None
    actions = []
    img, targets = render()

    def target_at_pointer():
        if pointer is None:
            return None
        px = pointer[0] * img.shape[1] // display_size[0]
        py = pointer[1] * img.shape[0] // display_size[1]
        for (x, y, w, h), action in targets:
            if x <= px < x+w and y <= py < y+h:
                return action
        return None

    def mouse(event, x, y, flags, data):
        nonlocal pointer, hovered, pressed
        pointer = (x, y)
        hovered = target_at_pointer()
        if event == cv2.EVENT_LBUTTONDOWN:
            pressed = hovered
        elif event == cv2.EVENT_LBUTTONUP:
            if pressed is not None and pressed == hovered:
                actions.append(pressed)
            pressed = None

    cv2.setMouseCallback(WINDOW, mouse)
    try:
        last_state = None
        while True:
            index = animation[0]() if animation else None
            state = (hovered, pressed, index)
            if state != last_state:
                frame = img.copy()
                if animation:
                    _, frames, position = animation
                    tile = frames[index]
                    x, y = position
                    frame[y:y+tile.shape[0], x:x+tile.shape[1]] = tile
                for (x, y, w, h), action in targets:
                    if hovered == action:
                        down = pressed == action
                        region = frame[y:y+h, x:x+w]
                        tint = np.full_like(region, PINK if down else CYAN)
                        amount = .38 if down else .19
                        cv2.addWeighted(region, 1-amount, tint, amount, 0, dst=region)
                        cv2.rectangle(frame,(x+2,y+2),(x+w-2,y+h-2),WHITE if down else CYAN,3)
                show_fullscreen(frame)
                last_state = state
            key = cv2.waitKeyEx(16)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return
            if actions or key >= 0:
                yield actions.pop(0) if actions else key
                img, targets = render()
                hovered = target_at_pointer()
                last_state = None
    finally:
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow(WINDOW)


def menu(videos):
    sel = 0
    thumbs = {}
    def render():
        img = panel("DANCE LIBRARY")
        targets = []
        start = sel // 3 * 3
        for i in range(start, min(start+3, len(videos))):
            x = 36 + (i-start)*300
            if i not in thumbs:
                cap = cv2.VideoCapture(videos[i])
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)*.15))
                ok, frame = cap.read()
                cap.release()
                thumbs[i] = fit_height(frame, 180) if ok else None
            frame = thumbs[i]
            if frame is not None:
                width = min(280, frame.shape[1])
                offset = (frame.shape[1]-width)//2
                left = x+(280-width)//2
                img[160:340,left:left+width] = frame[:,offset:offset+width]
            cv2.rectangle(img,(x,158),(x+280,393), PINK if i == sel else CYAN, 3 if i == sel else 1)
            name = os.path.splitext(os.path.basename(videos[i]))[0]
            text(img, name[:23], (x+9,374), .55, WHITE, 1)
            targets.append(((x,158,280,235), ("card",i)))
        if videos:
            text(img, os.path.basename(videos[sel])[:70], (36,435), .55)
            for box,label,action in [((36,465,145,47),"Previous","prev"),((196,465,145,47),"Next","next"),((701,465,220,47),"PLAY   [ENTER]","play")]:
                control(img,box,label,action=="play")
                targets.append((box,action))
        else:
            text(img,"Add a dance video to the videos folder.",(36,260))
        return img, targets
    for key in screen_input(render):
        if key in (27,ord("q")): return None
        if not videos: continue
        if isinstance(key, tuple): sel = key[1]
        elif key in ("prev",81,82,65361,65362,2424832,2490368,ord("w"),ord("a")): sel=max(0,sel-1)
        elif key in ("next",83,84,65363,65364,2555904,2621440,ord("s"),ord("d")): sel=min(len(videos)-1,sel+1)
        elif key in ("play",13,10): return videos[sel]


def countdown(cam, mirror, ref_frame=None, seconds=3):
    fullscreen()
    end = time.time() + seconds
    while time.time() < end:
        ok, f = cam.read()
        if not ok:
            return False
        if mirror:
            f = cv2.flip(f, 1)
        f = fit_height(f, PANEL_H)
        n = int(end - time.time()) + 1

        cw = f.shape[1]
        nw = cv2.getTextSize(str(n), cv2.FONT_HERSHEY_SIMPLEX, 3.5, 7)[0][0]
        text(f, str(n), ((cw - nw) // 2, PANEL_H // 2 + 40), 3.5, (0, 255, 255), 7)
        text(f, "Step back so your whole body is visible!", (20, 40), 0.7, (255, 255, 255))
        text(f, "YOU", (15, PANEL_H - 15), 0.65, (200, 200, 200), 1)
        preview = f

        if ref_frame is not None:
            rf = fit_height(ref_frame.copy(), PANEL_H)
            rw = rf.shape[1]
            ready_txt = f"READY: {n}"
            rtw = cv2.getTextSize(ready_txt, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0][0]
            text(rf, ready_txt, ((rw - rtw) // 2, PANEL_H // 2 + 20), 1.2, (0, 255, 255), 3)
            text(rf, "DANCER", (15, PANEL_H - 15), 0.65, (200, 200, 200), 1)
            preview = np.hstack((rf, f))
        show_fullscreen(preview)

        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord("q")):
            return False
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return False
    return True


def play(path, cam, mirror):
    ref_vecs, ref_valid, ref_weights, fps = analyse_reference(path, None)
    ref = cv2.VideoCapture(path)
    win = max(1, int(SYNC_WINDOW_SEC * fps))

    cv2.destroyAllWindows()

    ok, first_ref = ref.read()
    if not ok:
        ref.release()
        return [], 0

    if not countdown(cam, mirror, first_ref):
        ref.release()
        cv2.destroyAllWindows()
        return [], 0

    ref.set(cv2.CAP_PROP_POS_FRAMES, 0)

    audio = Audio(path)
    start = time.time()
    ref_pos, ref_frame = -1, None
    scores = []
    combo = best_combo = 0
    shown_score, label = 0.0, ("", (255, 255, 255))
    err, matched = None, None
    user_history = []
    pose_history = []
    alignment_checks = 0
    lag_frames = None

    with mp_pose.Pose(model_complexity=1, min_detection_confidence=0.4,
                      min_tracking_confidence=0.4) as pose:
        while True:
            t = time.time() - start
            target = int(t * fps)
            # advance reference video to the current time
            ended = False
            while ref_pos < target:
                ok, fr = ref.read()
                if not ok:
                    ended = True
                    break
                ref_pos, ref_frame = ref_pos + 1, fr
            if ended:
                break

            ok, cf = cam.read()
            if not ok:
                break
            if mirror:
                cf = cv2.flip(cf, 1)

            res = pose.process(cv2.cvtColor(cf, cv2.COLOR_BGR2RGB))
            screen_lm, world_lm = extract_landmarks(res)
            cam_img = fit_height(cf, PANEL_H)

            active_weights = ref_weights[ref_pos] if (0 <= ref_pos < len(ref_weights)) else None

            if screen_lm is not None and ref_pos < len(ref_vecs):
                user_vecs, user_ok = segment_vectors(world_lm, screen_lm)
                pose_history.append((ref_pos, user_vecs.copy(), user_ok.copy()))
                pose_history = pose_history[-ALIGN_HISTORY:]
                alignment_checks += 1
                if lag_frames is None and len(pose_history) == ALIGN_HISTORY and alignment_checks % ALIGN_RECHECK == 0:
                    lag_frames = estimate_lag(pose_history, ref_vecs, ref_valid, ref_weights, fps)
                aligning = lag_frames is None
                center = ref_pos if aligning else int(round(ref_pos - lag_frames))
                match_res = None if aligning else compare(
                    user_vecs, user_ok, ref_vecs, ref_valid, ref_weights, center, win, fps)
                if match_res:
                    raw, err, matched, tag, dt = match_res
                    matched_ref_pos = int(np.clip(round(center + dt * fps), 0, len(ref_weights) - 1))
                    active_weights = ref_weights[matched_ref_pos]
                    # Track motion over a longer interval so detector jitter does not count
                    # as dancing. If the reference is moving and the user is still, score 0.
                    user_history.append(screen_lm.copy())
                    user_history = user_history[-STILL_WINDOW:]
                    expected = active_weights
                    if len(user_history) >= STILL_WINDOW and expected is not None:
                        motion = body_motion(screen_lm, user_history[0])
                        expected_active = expected > 1.6
                        # Establish the choreography offset once the user actually moves.
                        # Thereafter follow it gradually, preventing frame-by-frame pose cherry-picking.
                        if motion is not None and motion >= STILL_BODY_MOTION and lag_frames is not None:
                            lag_adjustment = float(np.clip(dt, -0.12, 0.12) * fps * 0.12)
                            lag_frames = float(np.clip(lag_frames - lag_adjustment,
                                                       -MAX_LAG_SEC * fps,
                                                       MAX_LAG_SEC * fps))
                        if expected_active.any() and motion is not None:
                            if motion < STILL_BODY_MOTION:
                                raw = 0.0
                            else:
                                raw *= float(np.clip(motion / (STILL_BODY_MOTION * 2.5), 0.0, 1.0))
                    if lag_frames is None:
                        label = ("ALIGNING", (0, 220, 255))
                        combo = 0
                    else:
                        # Session score is hit rate: each evaluated pose earns one point
                        # at 50% similarity, rather than averaging similarity magnitudes.
                        hit = raw >= HIT_THRESHOLD
                        point_score = 100.0 if hit else 0.0
                        scores.append(point_score)
                        shown_score += (point_score - shown_score) * 0.25
                        label = judgement(raw, tag)
                        combo = combo + 1 if hit else 0
                        best_combo = max(best_combo, combo)
                else:
                    if lag_frames is None:
                        label = ("ALIGNING", (0, 220, 255))
                    else:
                        scores.append(0.0)
                        combo = 0
                        shown_score += (0.0 - shown_score) * 0.25
                        label = judgement(0.0)
                # Keep showing detected landmarks while phrase alignment is being established.
                draw_user_skeleton(cam_img, screen_lm, err, matched, active_weights)
            else:
                # Count missed tracking as misses; excluding these frames inflates the final grade.
                if screen_lm is not None and lag_frames is None:
                    label = ("ALIGNING", (0, 220, 255))
                else:
                    scores.append(0.0)
                    combo = 0
                    shown_score += (0.0 - shown_score) * 0.25
                    label = judgement(0.0)
                if screen_lm is None:
                    text(cam_img, "No body detected - step back", (20, 95), 0.7, (80, 80, 255))

            if ref_frame is not None:
                ref_img = fit_height(ref_frame, PANEL_H)
                avg = sum(scores) / len(scores) if scores else 0
                draw_hud(ref_img, "DANCER", avg, label, combo, shown_score)
                draw_hud(cam_img, "YOU", avg, label, combo, shown_score)
                show_fullscreen(np.hstack((ref_img, cam_img)))

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

    audio.stop()
    ref.release()
    cv2.destroyAllWindows()
    return scores, best_combo


# Angular pen paths in glyph coordinates, with individual metal finishes (BGR).
RANK_GLYPHS = {
    "S": ((32, 174, 239), [[(79,12),(39,8),(18,26),(20,48),(64,64),(66,81),(45,103),(9,96)]]),
    "A": ((190, 187, 179), [[(7,105),(37,13),(54,6),(75,100)], [(21,69),(66,64)]]),
    "B": ((217, 105, 37), [[(17,103),(17,18),(57,7),(76,19),(72,42),(22,59),(64,48),(78,61),(76,85),(48,102),(17,95)]]),
    "C": ((79, 165, 44), [[(78,16),(52,7),(20,24),(15,82),(38,102),(74,87)]]),
    "D": ((51, 43, 199), [[(17,104),(17,17),(47,8),(75,28),(75,80),(48,99),(17,93)]]),
    "E": ((181, 63, 146), [[(76,9),(19,19),(17,99),(72,91)],[(20,59),(62,50)]]),
    "F": ((35, 32, 30), [[(17,104),(19,19),(77,8)],[(18,59),(63,50)]]),
}


def draw_rank(img, letter):
    color, strokes = RANK_GLYPHS[letter]
    mask = np.zeros((142, 142), np.uint8)
    # A broad diagonal chisel gives each stroke a pointed terminal.
    nib = np.array([[-10,6],[-5,-4],[11,-9],[6,4]])
    for stroke in strokes:
        points = np.array([(x+25+(105-y)//10,y+16) for x,y in stroke])
        for start, end in zip(points, points[1:]):
            hull = cv2.convexHull(np.int32(np.vstack((start+nib,end+nib))))
            cv2.fillConvexPoly(mask,hull,255,lineType=cv2.LINE_AA)
    region = img[177:319,382:524]
    rim = cv2.dilate(mask,np.ones((3,3),np.uint8)) > 90
    region[rim] = (143,138,132)
    alpha = mask[:,:,None]/255.
    region[:] = region*(1-alpha)+np.array(color)*alpha


def result_screen(scores, best_combo):
    avg = sum(scores) / len(scores) if scores else 0
    g = grade(avg)
    with Image.open(os.path.join(HERE, "assets", "canvas.png")) as source:
        canvas = ImageOps.fit(source.convert("RGB"), (960, 540))
    img = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
    frames, durations = [], []
    position = (24, 24)
    with Image.open(os.path.join(HERE, "assets", "giphy.gif")) as gif:
        for page in ImageSequence.Iterator(gif):
            tile = page.convert("RGBA")
            tile.thumbnail((320, 397), Image.Resampling.LANCZOS)
            background = canvas.crop((24, 24, 24+tile.width, 24+tile.height))
            background.paste(tile, mask=tile.getchannel("A"))
            frames.append(cv2.cvtColor(np.asarray(background), cv2.COLOR_RGB2BGR))
            durations.append(max(10, page.info.get("duration", 100)))
    ends = np.cumsum(durations)
    started = time.monotonic()
    def current_frame():
        elapsed = (time.monotonic()-started)*1000 % ends[-1]
        return int(np.searchsorted(ends, elapsed, side="right"))

    text(img, "SESSION RESULTS", (388,92), 1.35, GOLD, 3)
    text(img,"GRADE",(392,171),.7,PINK)
    draw_rank(img, g)
    text(img,f"HIT RATE     {avg:.1f}%",(392,351),.95,CYAN)
    text(img,f"BEST COMBO   {best_combo}",(392,411),.95,PINK)
    targets = []
    for i,(action,key) in enumerate([("retry","r"),("menu","m"),("quit","q")]):
        box = (36+i*300,465,278,47)
        control(img,box,f"{action.upper()}   [{key.upper()}]",i==0)
        targets.append((box,action))
    for key in screen_input(lambda: (img,targets), (current_frame, frames, position)):
        if key in ("retry",ord("r")): return "retry"
        if key in ("menu",ord("m")): return "menu"
        if key in ("quit",ord("q"),27): break
    return "quit"


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
