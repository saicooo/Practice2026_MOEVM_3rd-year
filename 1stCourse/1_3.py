# main.py
import numpy as np


class VerticalController:
    def __init__(self, p_gain, d_gain, low_limit=None, high_limit=None):
        self.p_gain = p_gain
        self.d_gain = d_gain
        self.low_limit = low_limit
        self.high_limit = high_limit

    def calculate(self, desired, current, velocity):
        difference = desired - current

        force = (
            self.p_gain * difference
            - self.d_gain * velocity
        )

        if self.low_limit is not None:
            force = max(self.low_limit, force)

        if self.high_limit is not None:
            force = min(self.high_limit, force)

        return force


class MotionController:
    def __init__(self, p, v, a, j, limit):
        self.p = p
        self.v = v
        self.a = a
        self.j = j

        self.limit = limit

        self.previous_velocity = 0.0
        self.previous_acceleration = 0.0

    def calculate(self, target, position, velocity):
        position_error = target - position

        acceleration = velocity - self.previous_velocity
        jerk = acceleration - self.previous_acceleration

        self.previous_velocity = velocity
        self.previous_acceleration = acceleration

        result = (
            (self.p * position_error)
            - (self.v * velocity)
            - (self.a * acceleration)
            - (self.j * jerk)
        )

        result = min(self.limit, result)
        result = max(-self.limit, result)

        return result


class DroneCore:
    DEFAULT_HOVER_SPEED = 14468.429

    def __init__(self, simulation_callback):
        self.simulation_callback = simulation_callback

        self.coordinates = np.zeros(3)
        self.speeds = np.zeros(3)

        self.engine_scheme = [
            (-1, 1),
            (1, 1),
            (1, -1),
            (-1, -1)
        ]

    def send_motor_signal(self, x_force, y_force, z_force):
        engines = []

        for x_mix, y_mix in self.engine_scheme:
            motor_speed = (
                self.DEFAULT_HOVER_SPEED
                + z_force
                + (x_mix * x_force)
                + (y_mix * y_force)
            )

            engines.append(max(0.0, motor_speed))

        new_position, new_speed = self.simulation_callback(
            np.array([engines])
        )

        self.coordinates = new_position.flatten()
        self.speeds = new_speed.flatten()


class TargetPoint:
    def __init__(self, x, y, z, wait_steps=20):
        self.position = np.array([x, y, z])
        self.wait_steps = wait_steps
        self.current_wait = 0

    def process(self, drone, controllers):
        x_control = controllers["x"].calculate(
            self.position[0],
            drone.coordinates[0],
            drone.speeds[0]
        )

        y_control = controllers["y"].calculate(
            self.position[1],
            drone.coordinates[1],
            drone.speeds[1]
        )

        z_control = controllers["z"].calculate(
            self.position[2],
            drone.coordinates[2],
            drone.speeds[2]
        )

        drone.send_motor_signal(
            x_control,
            y_control,
            z_control
        )

        distance = np.linalg.norm(
            self.position - drone.coordinates
        )

        if distance < 0.2:
            self.current_wait += 1
        else:
            self.current_wait = 0

        return self.current_wait >= self.wait_steps


class AutoNavigation:
    def __init__(self, drone, route):
        self.drone = drone
        self.route = route
        self.route_index = 0

        self.controllers = {
            "x": MotionController(
                p=40.0,
                v=120.0,
                a=6000.0,
                j=170000.0,
                limit=400.0
            ),
            "y": MotionController(
                p=40.0,
                v=120.0,
                a=6000.0,
                j=170000.0,
                limit=400.0
            ),
            "z": VerticalController(
                p_gain=5000.0,
                d_gain=2000.0,
                low_limit=-2500.0,
                high_limit=2000.0
            )
        }

    def update(self):
        if self.route_index >= len(self.route):
            return True

        current_target = self.route[self.route_index]

        if current_target.process(
            self.drone,
            self.controllers
        ):
            self.route_index += 1

        return False


def controller(control_drones):
    drone = DroneCore(control_drones)

    path = [
        TargetPoint(0.0, 0.0, 2.0, wait_steps=8),
        TargetPoint(0.0, 2.0, 2.0, wait_steps=8),
        TargetPoint(5.0, 2.0, 2.0, wait_steps=8),
        TargetPoint(5.0, 2.0, 0.0, wait_steps=8)
    ]

    navigation_system = AutoNavigation(
        drone,
        path
    )

    completed = False

    while not completed:
        completed = navigation_system.update()