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
SYNC_WINDOW_SEC = 0.12   # forgiveness for timing search (+/- seconds)
VIS_MIN = 0.5

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
        elif abs_dt <= 0.08:
            timing_mult = 0.82
            tag = "LATE" if delta_t < 0 else "EARLY"
        elif abs_dt <= 0.12:
            timing_mult = 0.60
            tag = "TOO LATE" if delta_t < 0 else "TOO EARLY"
        else:
            timing_mult = 0.40
            tag = "OFF BEAT"

        final_score = pose_score * timing_mult

        if best is None or final_score > best[0]:
            best = (final_score, err, both, tag, delta_t)

    return best


# ------------------------------------------------------ reference analysis
def analyse_reference(path, canvas):
    """Run 3D pose detection and motion velocity analysis on the reference video (cached)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = f"{os.path.basename(path)}_{os.path.getsize(path)}_rhythm_v1.npz"
    cache = os.path.join(CACHE_DIR, key)
    if os.path.exists(cache):
        d = np.load(cache)
        return d["vecs"], d["valid"], d["weights"], float(d["fps"])

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    vecs, valid = [], []

    with mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
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
                cv2.imshow(WINDOW, img)
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
    if score >= 55:
        return "OK", (255, 200, 0)
    if score >= 35:
        return "BAD", (80, 120, 255)
    return "MISS", (60, 60, 255)


def grade(avg):
    for g, t in (("S", 92), ("A", 80), ("B", 65), ("C", 50), ("D", 35)):
        if avg >= t:
            return g
    return "F"


def draw_hud(img, title, avg, label, combo, shown_score):
    h, w = img.shape[:2]
    # Header background bar
    cv2.rectangle(img, (0, 0), (w, 54), (25, 25, 25), -1)
    # Average score on left
    text(img, f"SCORE {avg:5.1f}", (15, 36), 0.85, (0, 255, 255), 2)
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
        cv2.rectangle(img, (0, 54), (meter_w, 60), (0, 220, 120), -1)
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
def menu(videos):
    sel = 0
    while True:
        img = np.zeros((520, 900, 3), np.uint8)
        text(img, "DANCE COACH", (40, 70), 1.4, (0, 220, 255), 3)
        text(img, "Up/Down + Enter to choose   |   Q to quit", (40, 110), 0.6, (180, 180, 180), 1)
        if not videos:
            text(img, f"Put dance videos (.mp4) in: {VIDEO_DIR}", (40, 220), 0.7, (80, 80, 255))
        for i, v in enumerate(videos[:10]):
            col = (0, 255, 120) if i == sel else (220, 220, 220)
            text(img, ("> " if i == sel else "  ") + os.path.basename(v), (60, 170 + i * 34), 0.8, col)
        cv2.imshow(WINDOW, img)
        k = cv2.waitKeyEx(50)
        if k in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return None
        if k in (2490368, 82, ord("w")): sel = max(0, sel - 1)            # up
        if k in (2621440, 84, ord("s")): sel = min(len(videos) - 1, sel + 1)  # down
        if k in (13, 10) and videos:
            return videos[sel]


def countdown(cam, mirror, ref_frame=None, seconds=3):
    cv2.namedWindow(WIN_VIDEO, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_CAM, cv2.WINDOW_NORMAL)

    cam_w = 640
    ref_w = 640
    if ref_frame is not None:
        ref_w = int(ref_frame.shape[1] * (PANEL_H / ref_frame.shape[0]))
    cv2.resizeWindow(WIN_VIDEO, ref_w, PANEL_H)
    cv2.resizeWindow(WIN_CAM, cam_w, PANEL_H)
    cv2.moveWindow(WIN_VIDEO, 60, 60)
    cv2.moveWindow(WIN_CAM, 60 + ref_w + 16, 60)

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
        cv2.imshow(WIN_CAM, f)

        if ref_frame is not None:
            rf = fit_height(ref_frame.copy(), PANEL_H)
            rw = rf.shape[1]
            ready_txt = f"READY: {n}"
            rtw = cv2.getTextSize(ready_txt, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0][0]
            text(rf, ready_txt, ((rw - rtw) // 2, PANEL_H // 2 + 20), 1.2, (0, 255, 255), 3)
            text(rf, "DANCER", (15, PANEL_H - 15), 0.65, (200, 200, 200), 1)
            cv2.imshow(WIN_VIDEO, rf)

        k = cv2.waitKey(20) & 0xFF
        if k in (27, ord("q")):
            return False
        if cv2.getWindowProperty(WIN_CAM, cv2.WND_PROP_VISIBLE) < 1 or \
           (ref_frame is not None and cv2.getWindowProperty(WIN_VIDEO, cv2.WND_PROP_VISIBLE) < 1):
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
    scores, recent = [], []
    combo = best_combo = 0
    shown_score, label = 0.0, ("", (255, 255, 255))
    err, matched = None, None
    prev_user_vecs = None

    with mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
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

            if screen_lm is not None and ref_pos < len(ref_vecs) and ref_valid[ref_pos].sum() >= 4:
                user_vecs, user_ok = segment_vectors(world_lm, screen_lm)
                match_res = compare(user_vecs, user_ok, ref_vecs, ref_valid, ref_weights, ref_pos, win, fps)
                if match_res:
                    raw, err, matched, tag, dt = match_res
                    # Motion deficit: penalize if reference expects movement but user is still
                    if prev_user_vecs is not None and active_weights is not None:
                        dots_u = np.clip((user_vecs * prev_user_vecs).sum(axis=-1), -1.0, 1.0)
                        user_vel = np.degrees(np.arccos(dots_u))
                        expected_motion = np.clip((active_weights - 1.0) / 4.0, 0.0, 1.0)
                        actual_motion = np.clip(user_vel / 4.0, 0.0, 1.0)
                        motion_match = np.where(expected_motion > 0.2,
                                                np.clip(actual_motion / (expected_motion + 0.05), 0.0, 1.0),
                                                1.0)
                        motion_factor = 0.5 + 0.5 * motion_match[user_ok].mean()
                        raw = raw * motion_factor
                    prev_user_vecs = user_vecs.copy()
                    recent = (recent + [raw])[-3:]
                    smooth = sum(recent) / len(recent)
                    scores.append(smooth)
                    shown_score += (smooth - shown_score) * 0.25
                    label = judgement(smooth, tag)
                    combo = combo + 1 if smooth >= 75 else 0
                    best_combo = max(best_combo, combo)
                draw_user_skeleton(cam_img, screen_lm, err, matched, active_weights) if err is not None else None
            elif screen_lm is None:
                text(cam_img, "No body detected - step back", (20, 95), 0.7, (80, 80, 255))

            if ref_frame is not None:
                ref_img = fit_height(ref_frame, PANEL_H)
                avg = sum(scores) / len(scores) if scores else 0
                draw_hud(ref_img, "DANCER", avg, label, combo, shown_score)
                draw_hud(cam_img, "YOU", avg, label, combo, shown_score)
                cv2.imshow(WIN_VIDEO, ref_img)
                cv2.imshow(WIN_CAM, cam_img)

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
            if cv2.getWindowProperty(WIN_VIDEO, cv2.WND_PROP_VISIBLE) < 1 or \
               cv2.getWindowProperty(WIN_CAM, cv2.WND_PROP_VISIBLE) < 1:
                break

    audio.stop()
    ref.release()
    cv2.destroyAllWindows()
    return scores, best_combo


def result_screen(scores, best_combo):
    avg = sum(scores) / len(scores) if scores else 0
    g = grade(avg)
    while True:
        img = np.zeros((520, 900, 3), np.uint8)
        text(img, "RESULTS", (40, 80), 1.4, (0, 220, 255), 3)
        text(img, f"Grade: {g}", (40, 190), 2.6, (0, 255, 120), 5)
        text(img, f"Score: {avg:.1f} / 100", (40, 270), 1.0)
        text(img, f"Best combo: {best_combo}", (40, 320), 1.0)
        text(img, "R = retry     M = menu     Q = quit", (40, 450), 0.8, (180, 180, 180), 1)
        cv2.imshow(WINDOW, img)
        k = cv2.waitKey(50) & 0xFF
        if k == ord("r"):
            cv2.destroyAllWindows()
            return "retry"
        if k == ord("m"):
            cv2.destroyAllWindows()
            return "menu"
        if k in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
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
