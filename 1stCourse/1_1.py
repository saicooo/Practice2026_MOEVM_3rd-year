import numpy as np

def controller(control_drones):
    """
    Quadcopter controller for takeoff, waypoint passage and landing
    within 20 seconds, while keeping x,y inside (-4,4) and z in (0,4).
    Required path: (0,0,0.1) -> (0,0,2) -> (0,0,0.1).
    """
    hover_speed = 14468.429

    # Waypoints the drone must pass through (with radius 0.2 m)
    waypoints = [
        np.array([0.0, 0.0, 0.1]),  # first checkpoint just above ground
        np.array([0.0, 0.0, 2.0]),  # hover point
        np.array([0.0, 0.0, 0.1]),  # descent checkpoint
        np.array([0.0, 0.0, 0.0])   # final landing
    ]
    current_wp = 0
    wp_step_count = 0
    max_steps_per_wp = 300   # safety timeout per waypoint (~6 s at 50 Hz)

    # Vertical PD gains (tuned empirically)
    Kp_z = 500.0
    Kd_z = 300.0
    max_delta_z = 2000.0

    # Horizontal PD gains
    Kp_xy = 5.0
    Kd_xy = 3.0
    max_xy_ctrl = 200.0

    # Initial action: all rotors at hover speed
    action = np.array([[hover_speed, hover_speed, hover_speed, hover_speed]])

    while True:
        # Get current state from simulator
        pos, vel = control_drones(action)
        x, y, z = pos[0]
        vx, vy, vz = vel[0]

        # Current target
        target = waypoints[current_wp]
        tx, ty, tz = target

        # Distance to current waypoint
        dist = np.linalg.norm([x - tx, y - ty, z - tz])

        # Advance to next waypoint if close enough or stuck
        if dist < 0.15 or wp_step_count > max_steps_per_wp:
            if current_wp < len(waypoints) - 1:
                current_wp += 1
                wp_step_count = 0
                target = waypoints[current_wp]
                tx, ty, tz = target

        # ----- Vertical (altitude) control -----
        error_z = tz - z
        delta_z = Kp_z * error_z - Kd_z * vz   # desired vz = 0
        delta_z = np.clip(delta_z, -max_delta_z, max_delta_z)

        # ----- Horizontal (x, y) control -----
        # x correction (pitch)
        error_x = tx - x
        ctrl_x = Kp_xy * error_x - Kd_xy * vx
        ctrl_x = np.clip(ctrl_x, -max_xy_ctrl, max_xy_ctrl)

        # y correction (roll)
        error_y = ty - y
        ctrl_y = Kp_xy * error_y - Kd_xy * vy
        ctrl_y = np.clip(ctrl_y, -max_xy_ctrl, max_xy_ctrl)

        # ----- Motor mixing for a square ("X") configuration -----
        # Rotor layout (top view, z out of screen):
        #   2 (rear left)    3 (rear right)
        #   1 (front left)   0 (front right)
        # x‑mix gives pitch torque, y‑mix gives roll torque.
        # Signs are chosen so that positive ctrl_x moves the drone in +x,
        # positive ctrl_y moves it in +y.
        base = hover_speed + delta_z
        action[0, 0] = base + ctrl_x - ctrl_y   # motor 0
        action[0, 1] = base - ctrl_x - ctrl_y   # motor 1
        action[0, 2] = base - ctrl_x + ctrl_y   # motor 2
        action[0, 3] = base + ctrl_x + ctrl_y   # motor 3

        # Keep rotor speeds within a safe range
        action = np.clip(action, 0, 30000)

        wp_step_count += 1