import numpy as np
from PIL import Image
import cv2


def build_color_masks(rgb_img):
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    red_low = cv2.inRange(hsv_img, np.array([0, 100, 100]), np.array([10, 255, 255]))
    red_high = cv2.inRange(hsv_img, np.array([160, 100, 100]), np.array([180, 255, 255]))
    red_full = cv2.bitwise_or(red_low, red_high)
    blue_full = cv2.inRange(hsv_img, np.array([100, 100, 50]), np.array([140, 255, 255]))
    return red_full, blue_full


def controller(control_drones):
    v_cmd = np.array([0.0, 0.0, 0.0])
    state = "HOVER"
    ok_frames = 0
    heading = None
    px_to_m = 0.01

    while True:
        img: Image.Image = control_drones([v_cmd])[0]
        rgb = np.array(img)

        red_mask, blue_mask = build_color_masks(rgb)
        h, w = rgb.shape[:2]
        cy, cx = h / 2.0, w / 2.0

        red_cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        chosen_cnt = None
        chosen_score = -np.inf
        chosen_cx, chosen_cy = cx, cy
        chosen_area = 0

        for cnt in red_cnts:
            a = cv2.contourArea(cnt)
            if a > 400:
                mu = cv2.moments(cnt)
                if mu["m00"] > 0:
                    rcx = mu["m10"] / mu["m00"]
                    rcy = mu["m01"] / mu["m00"]

                    dx = rcx - cx
                    dy = cy - rcy

                    if state == "FIND_SQUARE" and heading is not None:
                        proj = dx * heading[0] + dy * heading[1]
                        if proj < 0:
                            continue
                        score = a
                    elif state in ["HOVER", "LEAVE"]:
                        dist = np.hypot(dx, dy)
                        score = -dist
                    else:
                        score = a

                    if score > chosen_score:
                        chosen_score = score
                        chosen_cnt = cnt
                        chosen_cx, chosen_cy = rcx, rcy
                        chosen_area = a

        if state == "FIND_SQUARE":
            if chosen_cnt is not None:
                state = "HOVER"
                ok_frames = 0

        if state == "HOVER":
            if chosen_cnt is not None:
                px_to_m = np.clip(1.0 / np.sqrt(chosen_area), 0.001, 0.1)
                err_px_x = chosen_cx - cx
                err_px_y = cy - chosen_cy
                err_m_x = err_px_x * px_to_m
                err_m_y = err_px_y * px_to_m

                if abs(err_m_x) < 0.25 and abs(err_m_y) < 0.25:
                    ok_frames += 1
                    v_cmd = np.array([0.0, 0.0, 0.0])
                else:
                    ok_frames = 0
                    v_cmd = np.array(
                        [
                            np.clip(err_m_x * 0.8, -1.5, 1.5),
                            np.clip(err_m_y * 0.8, -1.5, 1.5),
                            0.0,
                        ]
                    )

                if ok_frames >= 4:
                    state = "LEAVE"
                    b_rows, b_cols = np.where(blue_mask > 0)
                    if len(b_cols) > 50:
                        dx_b = b_cols - cx
                        dy_b = cy - b_rows
                        b_norm = np.hypot(dx_b, dy_b)

                        keep = b_norm > 20
                        dx_b = dx_b[keep]
                        dy_b = dy_b[keep]
                        b_norm = b_norm[keep]

                        if len(dx_b) > 10:
                            ux = dx_b / b_norm
                            uy = dy_b / b_norm
                            if heading is not None:
                                back = -heading
                                front_mask = (ux * back[0] + uy * back[1]) < 0.7
                                if np.sum(front_mask) > 10:
                                    mean_dx = np.mean(dx_b[front_mask])
                                    mean_dy = np.mean(dy_b[front_mask])
                                else:
                                    mean_dx, mean_dy = heading[0], heading[1]
                            else:
                                mean_dx = np.mean(dx_b)
                                mean_dy = np.mean(dy_b)

                            mean_n = np.hypot(mean_dx, mean_dy)
                            if mean_n > 0:
                                heading = np.array([mean_dx / mean_n, mean_dy / mean_n])
                            else:
                                heading = np.array([1.0, 0.0])
            else:
                ok_frames = 0
                v_cmd = np.array([0.0, 0.0, 0.0])

        elif state == "LEAVE":
            if heading is not None:
                v_cmd = np.array([heading[0] * 1.5, heading[1] * 1.5, 0.0])
            else:
                v_cmd = np.array([1.5, 0.0, 0.0])
            state = "FIND_SQUARE"

        elif state == "FIND_SQUARE":
            b_rows, b_cols = np.where(blue_mask > 0)
            if len(b_cols) > 50:
                dx_b = b_cols - cx
                dy_b = cy - b_rows
                if heading is not None:
                    n = np.clip(np.hypot(dx_b, dy_b), 1e-5, None)
                    ahead = (dx_b * heading[0] + dy_b * heading[1]) / n > -0.2
                else:
                    ahead = np.ones_like(dx_b, dtype=bool)

                if np.sum(ahead) > 20:
                    tgt_px_x = np.mean(dx_b[ahead])
                    tgt_px_y = np.mean(dy_b[ahead])
                    tgt_m_x = tgt_px_x * px_to_m
                    tgt_m_y = tgt_px_y * px_to_m
                    tgt_n = np.hypot(tgt_m_x, tgt_m_y)
                    if tgt_n > 0:
                        v_cmd = np.array([(tgt_m_x / tgt_n) * 1.5, (tgt_m_y / tgt_n) * 1.5, 0.0])
                        new_head = np.array([tgt_m_x / tgt_n, tgt_m_y / tgt_n])
                        if heading is not None:
                            heading = 0.8 * heading + 0.2 * new_head
                            heading /= np.hypot(heading[0], heading[1])
                        else:
                            heading = new_head
                    else:
                        if heading is not None:
                            v_cmd = np.array([heading[0] * 1.5, heading[1] * 1.5, 0.0])
                        else:
                            v_cmd = np.array([0.0, 0.0, 0.0])
                else:
                    if heading is not None:
                        v_cmd = np.array([heading[0] * 1.5, heading[1] * 1.5, 0.0])
                    else:
                        v_cmd = np.array([0.0, 0.0, 0.0])
            else:
                if heading is not None:
                    v_cmd = np.array([heading[0] * 1.5, heading[1] * 1.5, 0.0])
                else:
                    v_cmd = np.array([0.0, 0.0, 0.0])