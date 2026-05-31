import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image


# directions (image-space unit vectors)
DIR_VEC = {
    "right": np.array([1.0, 0.0], dtype=np.float32),
    "down": np.array([0.0, 1.0], dtype=np.float32),
    "left": np.array([-1.0, 0.0], dtype=np.float32),
    "up": np.array([0.0, -1.0], dtype=np.float32),
}

# heuristic: when we see a "corner" (two blue directions), pick the next direction to follow
CORNER_NEXT = {
    frozenset(("right", "down")): "right",
    frozenset(("left", "down")): "down",
    frozenset(("left", "up")): "left",
    frozenset(("right", "up")): "up",
}

# clockwise rotation of a chosen direction
TURN_RIGHT = {
    "right": "down",
    "down": "left",
    "left": "up",
    "up": "right",
}

N_TARGET = 4
N_HOLD = 2
CENTER_EPS = 0.23
STEP_FWD = 0.95
CAP_MOVE = 1.15
CAP_CENTER = 0.55


@dataclass
class AgentMem:
    phase: str = "hold"
    hold_t: int = N_HOLD
    n_visits: int = 0
    dir_name: Optional[str] = None
    m_per_px: Optional[float] = None
    side_px: Optional[float] = None
    img_to_xy: np.ndarray = field(default_factory=lambda: np.eye(2, dtype=np.float32))
    n_steps: int = 0
    left_flag: bool = False
    arrived_cnt: int = 0
    done: bool = False


def controller(control_drones):
    mems = [AgentMem() for _ in range(4)]
    imgs = do_calibration(control_drones, mems)

    while True:
        out = np.zeros((4, 3), dtype=np.float32)

        for k in range(4):
            obs = parse_frame(imgs[k], mems[k].m_per_px, mems[k].side_px)
            update_scale(mems[k], obs)
            out[k] = decide(mems[k], obs)

        imgs = control_drones(out)


def do_calibration(control_drones, mems):
    z = np.zeros((4, 3), dtype=np.float32)
    imgs0 = control_drones(z)

    probe = 0.22
    ax = np.zeros((4, 3), dtype=np.float32)
    ax[:, 0] = probe
    imgsx = control_drones(ax)

    ay = np.zeros((4, 3), dtype=np.float32)
    ay[:, 0] = -probe
    ay[:, 1] = probe
    imgsy = control_drones(ay)

    back = np.zeros((4, 3), dtype=np.float32)
    back[:, 1] = -probe
    imgs = control_drones(back)

    o0 = [parse_frame(im, None, None) for im in imgs0]
    ox = [parse_frame(im, o0[i]["scale"], o0[i]["side_px"]) for i, im in enumerate(imgsx)]
    oy = [parse_frame(im, o0[i]["scale"], o0[i]["side_px"]) for i, im in enumerate(imgsy)]

    for i in range(4):
        update_scale(mems[i], o0[i])
        q0 = red_offset_m(o0[i])
        qx = red_offset_m(ox[i])
        qy = red_offset_m(oy[i])

        if q0 is None or qx is None or qy is None:
            continue

        c_x = -(qx - q0) / probe
        c_y = -(qy - q0) / probe
        mat = np.column_stack((c_x, c_y)).astype(np.float32)

        det = float(np.linalg.det(mat))
        if abs(det) > 0.15 and np.all(np.isfinite(mat)):
            mems[i].img_to_xy = np.linalg.inv(mat).astype(np.float32)

    return imgs


def parse_frame(image, fallback_scale=None, fallback_side_px=None):
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    mid = np.array([w * 0.5, h * 0.5], dtype=np.float32)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    r1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([12, 255, 255]))
    r2 = cv2.inRange(hsv, np.array([168, 70, 50]), np.array([179, 255, 255]))
    mask_r = cv2.bitwise_or(r1, r2)

    mask_b = cv2.inRange(hsv, np.array([85, 45, 40]), np.array([135, 255, 255]))

    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, k3)
    mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_CLOSE, k5)
    mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, k3)
    mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_CLOSE, k5)

    reds = extract_red(mask_r, w, h)

    scale = fallback_scale
    side_px = fallback_side_px
    got_scale = False

    good = [r for r in reds if r["ratio"] <= 1.9 and r["side_px"] >= 8.0]
    if good:
        ref = max(good, key=lambda r: r["area"])
        side_px = ref["side_px"]
        scale = 1.0 / side_px
        got_scale = True
    elif reds and scale is None:
        ref = max(reds, key=lambda r: r["area"])
        side_px = ref["side_px"]
        scale = 1.0 / max(side_px, 1.0)
        got_scale = True

    if scale is None:
        side_px = float(min(w, h)) / 6.0
        scale = 1.0 / side_px

    if side_px is None:
        side_px = 1.0 / scale

    red_main = None
    if reds:
        red_main = min(reds, key=lambda r: float(np.linalg.norm(r["center"] - mid)))

    dirs, counts = blue_dirs(mask_b, red_main["center"] if red_main else mid, side_px)

    return {
        "w": w,
        "h": h,
        "center": mid,
        "scale": float(scale),
        "side_px": float(side_px),
        "scale_found": got_scale,
        "reds": reds,
        "red_center": red_main,
        "blue_mask": mask_b,
        "dirs": dirs,
        "blue_counts": counts,
    }


def extract_red(mask_r, w, h):
    contours, _ = cv2.findContours(mask_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(20.0, w * h * 0.00015)
    out = []

    for c in contours:
        a = float(cv2.contourArea(c))
        if a < min_area:
            continue

        (cx, cy), (rw, rh), _ = cv2.minAreaRect(c)
        if rw < 3.0 or rh < 3.0:
            continue

        ratio = max(rw, rh) / max(min(rw, rh), 1.0)
        side_px = math.sqrt(max(a, 1.0))
        out.append(
            {
                "center": np.array([cx, cy], dtype=np.float32),
                "side_px": float(side_px),
                "area": a,
                "ratio": float(ratio),
            }
        )

    return out


def blue_dirs(mask_b, anchor, side_px):
    ys, xs = np.nonzero(mask_b)
    counts = {name: 0 for name in DIR_VEC}

    if len(xs) == 0:
        return [], counts

    dx = xs.astype(np.float32) - float(anchor[0])
    dy = ys.astype(np.float32) - float(anchor[1])
    s = max(float(side_px), 8.0)

    min_d = 0.35 * s
    max_d = 4.5 * s
    band = 0.72 * s

    masks = {
        "right": (dx > min_d) & (dx < max_d) & (np.abs(dy) < band),
        "left": (dx < -min_d) & (dx > -max_d) & (np.abs(dy) < band),
        "down": (dy > min_d) & (dy < max_d) & (np.abs(dx) < band),
        "up": (dy < -min_d) & (dy > -max_d) & (np.abs(dx) < band),
    }

    for name, mask in masks.items():
        counts[name] = int(np.count_nonzero(mask))

    thresh = max(18, int(s * s * 0.025))
    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    dirs = [name for name, value in ordered if value >= thresh]

    return dirs[:2], counts


def update_scale(mem, obs):
    if obs["scale_found"]:
        mem.m_per_px = obs["scale"]
        mem.side_px = obs["side_px"]


def decide(mem, obs):
    if mem.done:
        return to_action(keep_center(mem, obs, 0.25))

    if mem.phase == "hold":
        if not is_centered(obs):
            return to_action(keep_center(mem, obs, CAP_CENTER))

        if mem.hold_t > 0:
            mem.hold_t -= 1

        if mem.hold_t == 0:
            mem.n_visits += 1
            if mem.n_visits >= N_TARGET:
                mem.done = True
            else:
                if mem.dir_name is None:
                    mem.dir_name = dir_from_corner(obs["dirs"]) or pick_strong_blue(obs) or "right"
                else:
                    mem.dir_name = TURN_RIGHT.get(mem.dir_name, dir_from_corner(obs["dirs"]) or "right")

                mem.phase = "move"
                mem.n_steps = 0
                mem.left_flag = False
                mem.arrived_cnt = 0

        return np.zeros(3, dtype=np.float32)

    if mem.dir_name is None:
        mem.dir_name = dir_from_corner(obs["dirs"]) or pick_strong_blue(obs) or "right"

    centered_now = is_centered(obs)
    if not centered_now:
        mem.left_flag = True

    if mem.left_flag and centered_now and mem.n_steps >= 1 and len(obs["dirs"]) >= 2:
        mem.arrived_cnt += 1
    else:
        mem.arrived_cnt = 0

    if mem.arrived_cnt >= 1:
        mem.phase = "hold"
        mem.hold_t = N_HOLD
        mem.n_steps = 0
        mem.left_flag = False
        mem.arrived_cnt = 0
        return to_action(keep_center(mem, obs, 0.35))

    xy = move_cmd(mem, obs)
    mem.n_steps += 1
    return to_action(xy)


def dir_from_corner(dirs):
    s = frozenset(dirs)
    if s in CORNER_NEXT:
        return CORNER_NEXT[s]

    for a in dirs:
        for b in dirs:
            if a != b:
                key = frozenset((a, b))
                if key in CORNER_NEXT:
                    return CORNER_NEXT[key]

    return None


def pick_strong_blue(obs):
    counts = obs["blue_counts"]
    if not counts:
        return None
    name, value = max(counts.items(), key=lambda item: item[1])
    return name if value > 0 else None


def is_centered(obs):
    off = red_offset_m(obs)
    if off is None:
        return False
    return abs(float(off[0])) <= CENTER_EPS and abs(float(off[1])) <= CENTER_EPS


def red_offset_m(obs):
    red = obs["red_center"]
    if red is None:
        return None
    return (red["center"] - obs["center"]) * obs["scale"]


def keep_center(mem, obs, max_norm):
    off = red_offset_m(obs)
    if off is None:
        return np.zeros(2, dtype=np.float32)

    xy = mem.img_to_xy @ off.astype(np.float32)
    return cap_xy(xy, max_norm)


def move_cmd(mem, obs):
    tgt = target_red(mem, obs)
    if tgt is not None:
        err_img = (tgt["center"] - obs["center"]) * obs["scale"]
        xy = mem.img_to_xy @ err_img.astype(np.float32)
        return cap_xy(xy, CAP_MOVE)

    v_img = DIR_VEC[mem.dir_name]
    n_img = np.array([-v_img[1], v_img[0]], dtype=np.float32)
    lat_m = corridor_lat_err(obs, v_img, n_img)

    desired = v_img * STEP_FWD + n_img * np.clip(lat_m * 0.8, -0.45, 0.45)
    xy = mem.img_to_xy @ desired.astype(np.float32)
    return cap_xy(xy, CAP_MOVE)


def target_red(mem, obs):
    if not obs["reds"] or mem.dir_name is None:
        return None

    v = DIR_VEC[mem.dir_name]
    center = obs["center"]
    side = max(float(obs["side_px"]), 8.0)
    cand = []

    for red in obs["reds"]:
        d = red["center"] - center
        proj = float(d @ v)
        dist = float(np.linalg.norm(d))

        if proj > 0.45 * side or (mem.left_flag and dist < 0.95 * side):
            cand.append((proj - dist * 0.12 + red["area"] * 0.0001, red))

    if not cand:
        return None

    return max(cand, key=lambda item: item[0])[1]


def corridor_lat_err(obs, v_img, n_img):
    blue = obs["blue_mask"]
    ys, xs = np.nonzero(blue)

    if len(xs) == 0:
        return 0.0

    coords = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    delta = coords - obs["center"]
    proj = delta @ v_img
    lat = delta @ n_img
    side = max(float(obs["side_px"]), 8.0)

    mask = (proj > 0.35 * side) & (proj < 5.0 * side) & (np.abs(lat) < 1.35 * side)
    if np.count_nonzero(mask) < 8:
        return 0.0

    return float(np.median(lat[mask]) * obs["scale"])


def cap_xy(xy, max_norm):
    xy = np.asarray(xy, dtype=np.float32)
    norm = float(np.linalg.norm(xy))
    if norm > max_norm:
        xy = xy / norm * max_norm
    return xy.astype(np.float32)


def to_action(xy):
    act = np.zeros(3, dtype=np.float32)
    act[:2] = xy
    return act