import numpy as np
from PIL import Image
import cv2

def build_color_filters(rgb_frame):
    hsv_view = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV)
    lower_red_a = cv2.inRange(hsv_view, np.array([0, 100, 100]), np.array([10, 255, 255]))
    lower_red_b = cv2.inRange(hsv_view, np.array([160, 100, 100]), np.array([180, 255, 255]))
    merged_red = cv2.bitwise_or(lower_red_a, lower_red_b)
    blue_zone = cv2.inRange(hsv_view, np.array([100, 100, 50]), np.array([140, 255, 255]))
    return merged_red, blue_zone

def controller(send_velocity_cmd):
    cmd_vector = np.array([0.0, 0.0, 0.0])
    operational_mode = 'STABILIZE'
    hover_timer = 0
    travel_heading = None
    px_to_m_ratio = 0.01

    while True:
        raw_image: Image.Image = send_velocity_cmd([cmd_vector])[0]
        image_data = np.array(raw_image)

        red_filter, blue_filter = build_color_filters(image_data)
        h_img, w_img = image_data.shape[:2]
        mid_y, mid_x = h_img / 2.0, w_img / 2.0

        red_edges, _ = cv2.findContours(red_filter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        chosen_blob = None
        top_score = -float('inf')
        blob_cx, blob_cy = mid_x, mid_y
        selected_area = 0

        for segment in red_edges:
            raw_area = cv2.contourArea(segment)
            if raw_area > 400:
                seg_moments = cv2.moments(segment)
                if seg_moments["m00"] > 0:
                    cand_cx = seg_moments["m10"] / seg_moments["m00"]
                    cand_cy = seg_moments["m01"] / seg_moments["m00"]

                    delta_px_x = cand_cx - mid_x
                    delta_px_y = mid_y - cand_cy

                    if operational_mode == 'SEEK_TARGET' and travel_heading is not None:
                        proj_ahead = delta_px_x * travel_heading[0] + delta_px_y * travel_heading[1]
                        if proj_ahead < 0:
                            continue
                        score_val = raw_area
                    elif operational_mode in ['STABILIZE', 'DEPART']:
                        dist_to_mid = np.hypot(delta_px_x, delta_px_y)
                        score_val = -dist_to_mid
                    else:
                        score_val = raw_area

                    if score_val > top_score:
                        top_score = score_val
                        chosen_blob = segment
                        blob_cx, blob_cy = cand_cx, cand_cy
                        selected_area = raw_area

        if operational_mode == 'SEEK_TARGET':
            if chosen_blob is not None:
                operational_mode = 'STABILIZE'
                hover_timer = 0

        if operational_mode == 'STABILIZE':
            if chosen_blob is not None:
                px_to_m_ratio = np.clip(1.0 / np.sqrt(selected_area), 0.001, 0.1)
                err_x_px = blob_cx - mid_x
                err_y_px = mid_y - blob_cy
                err_x_m = err_x_px * px_to_m_ratio
                err_y_m = err_y_px * px_to_m_ratio

                if abs(err_x_m) < 0.25 and abs(err_y_m) < 0.25:
                    hover_timer += 1
                    cmd_vector = np.array([0.0, 0.0, 0.0])
                else:
                    hover_timer = 0
                    cmd_vector = np.array([np.clip(err_x_m * 0.8, -1.5, 1.5),
                                           np.clip(err_y_m * 0.8, -1.5, 1.5), 0.0])

                if hover_timer >= 4:
                    operational_mode = 'DEPART'
                    blue_rows, blue_cols = np.where(blue_filter > 0)
                    if len(blue_cols) > 50:
                        diff_x = blue_cols - mid_x
                        diff_y = mid_y - blue_rows
                        distances = np.hypot(diff_x, diff_y)

                        valid_zone = distances > 20
                        diff_x = diff_x[valid_zone]
                        diff_y = diff_y[valid_zone]
                        distances = distances[valid_zone]

                        if len(diff_x) > 10:
                            unit_x = diff_x / distances
                            unit_y = diff_y / distances
                            if travel_heading is not None:
                                rear_vec = -travel_heading
                                front_mask = (unit_x * rear_vec[0] + unit_y * rear_vec[1]) < 0.7
                                if np.sum(front_mask) > 10:
                                    mean_dx = np.mean(diff_x[front_mask])
                                    mean_dy = np.mean(diff_y[front_mask])
                                else:
                                    mean_dx, mean_dy = travel_heading[0], travel_heading[1]
                            else:
                                mean_dx = np.mean(diff_x)
                                mean_dy = np.mean(diff_y)
                            norm_mean = np.hypot(mean_dx, mean_dy)
                            if norm_mean > 0:
                                travel_heading = np.array([mean_dx / norm_mean, mean_dy / norm_mean])
                            else:
                                travel_heading = np.array([1.0, 0.0])
            else:
                hover_timer = 0
                cmd_vector = np.array([0.0, 0.0, 0.0])

        elif operational_mode == 'DEPART':
            if travel_heading is not None:
                cmd_vector = np.array([travel_heading[0] * 1.5, travel_heading[1] * 1.5, 0.0])
            else:
                cmd_vector = np.array([1.5, 0.0, 0.0])
            operational_mode = 'SEEK_TARGET'

        elif operational_mode == 'SEEK_TARGET':
            blue_rows, blue_cols = np.where(blue_filter > 0)
            if len(blue_cols) > 50:
                diff_x = blue_cols - mid_x
                diff_y = mid_y - blue_rows
                if travel_heading is not None:
                    norms = np.clip(np.hypot(diff_x, diff_y), 1e-5, None)
                    front_mask = (diff_x * travel_heading[0] + diff_y * travel_heading[1]) / norms > -0.2
                else:
                    front_mask = np.ones_like(diff_x, dtype=bool)

                if np.sum(front_mask) > 20:
                    goal_px_x = np.mean(diff_x[front_mask])
                    goal_px_y = np.mean(diff_y[front_mask])
                    goal_m_x = goal_px_x * px_to_m_ratio
                    goal_m_y = goal_px_y * px_to_m_ratio
                    goal_norm = np.hypot(goal_m_x, goal_m_y)
                    if goal_norm > 0:
                        cmd_vector = np.array([(goal_m_x / goal_norm) * 1.5,
                                               (goal_m_y / goal_norm) * 1.5, 0.0])
                        fresh_dir = np.array([goal_m_x / goal_norm, goal_m_y / goal_norm])
                        if travel_heading is not None:
                            travel_heading = 0.8 * travel_heading + 0.2 * fresh_dir
                            travel_heading /= np.hypot(travel_heading[0], travel_heading[1])
                        else:
                            travel_heading = fresh_dir
                    else:
                        if travel_heading is not None:
                            cmd_vector = np.array([travel_heading[0] * 1.5, travel_heading[1] * 1.5, 0.0])
                        else:
                            cmd_vector = np.array([0.0, 0.0, 0.0])
                else:
                    if travel_heading is not None:
                        cmd_vector = np.array([travel_heading[0] * 1.5, travel_heading[1] * 1.5, 0.0])
                    else:
                        cmd_vector = np.array([0.0, 0.0, 0.0])
            else:
                if travel_heading is not None:
                    cmd_vector = np.array([travel_heading[0] * 1.5, travel_heading[1] * 1.5, 0.0])
                else:
                    cmd_vector = np.array([0.0, 0.0, 0.0])