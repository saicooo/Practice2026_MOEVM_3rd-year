# main.py
import numpy as np


class AxisPD:
    def __init__(self, proportional, derivative, minimum=None, maximum=None):
        self.proportional = proportional
        self.derivative = derivative
        self.minimum = minimum
        self.maximum = maximum

    def update(self, target, position, velocity):
        delta = target - position
        correction = (self.proportional * delta) - (self.derivative * velocity)

        if self.minimum is not None and correction < self.minimum:
            correction = self.minimum

        if self.maximum is not None and correction > self.maximum:
            correction = self.maximum

        return correction


class SmoothAxisController:
    def __init__(self, pos_gain, vel_gain, acc_gain, jerk_gain, limit):
        self.pos_gain = pos_gain
        self.vel_gain = vel_gain
        self.acc_gain = acc_gain
        self.jerk_gain = jerk_gain
        self.limit = limit

        self.prev_velocity = 0.0
        self.prev_acceleration = 0.0

    def update(self, target, current, velocity):
        position_error = target - current

        acceleration = velocity - self.prev_velocity
        jerk = acceleration - self.prev_acceleration

        self.prev_velocity = velocity
        self.prev_acceleration = acceleration

        value = (
            (self.pos_gain * position_error)
            - (self.vel_gain * velocity)
            - (self.acc_gain * acceleration)
            - (self.jerk_gain * jerk)
        )

        if value > self.limit:
            value = self.limit
        elif value < -self.limit:
            value = -self.limit

        return value


class Quadcopter:
    HOVER_SPEED = 14468.429

    def __init__(self, simulator_step):
        self.simulator_step = simulator_step

        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])

        self.motor_layout = [
            {"x": -1, "y": 1},
            {"x": 1, "y": 1},
            {"x": 1, "y": -1},
            {"x": -1, "y": -1}
        ]

    def move(self, control_x, control_y, control_z):
        motor_values = []

        for motor in self.motor_layout:
            force = (
                self.HOVER_SPEED
                + control_z
                + (motor["x"] * control_x)
                + (motor["y"] * control_y)
            )

            motor_values.append(max(0.0, force))

        coords, speeds = self.simulator_step(np.array([motor_values]))

        self.position = coords.flatten()
        self.velocity = speeds.flatten()


class RoutePoint:
    def __init__(self, px, py, pz, stable_steps=100):
        self.destination = np.array([px, py, pz])
        self.required_steps = stable_steps
        self.steps_inside_zone = 0

    def reached(self, drone_object, controllers):
        x_signal = controllers["x"].update(
            self.destination[0],
            drone_object.position[0],
            drone_object.velocity[0]
        )

        y_signal = controllers["y"].update(
            self.destination[1],
            drone_object.position[1],
            drone_object.velocity[1]
        )

        z_signal = controllers["z"].update(
            self.destination[2],
            drone_object.position[2],
            drone_object.velocity[2]
        )

        drone_object.move(x_signal, y_signal, z_signal)

        distance = np.linalg.norm(self.destination - drone_object.position)

        if distance < 0.2:
            self.steps_inside_zone += 1
        else:
            self.steps_inside_zone = 0

        return self.steps_inside_zone >= self.required_steps


class NavigationSystem:
    def __init__(self, quadcopter, checkpoints):
        self.quadcopter = quadcopter
        self.checkpoints = checkpoints
        self.current_checkpoint = 0

        self.controllers = {
            "x": SmoothAxisController(
                pos_gain=40.0,
                vel_gain=120.0,
                acc_gain=6000.0,
                jerk_gain=170000.0,
                limit=400.0
            ),
            "y": SmoothAxisController(
                pos_gain=40.0,
                vel_gain=120.0,
                acc_gain=6000.0,
                jerk_gain=170000.0,
                limit=400.0
            ),
            "z": AxisPD(
                proportional=5000.0,
                derivative=2000.0,
                minimum=-2500.0,
                maximum=2000.0
            )
        }

    def tick(self):
        if self.current_checkpoint >= len(self.checkpoints):
            return True

        active_point = self.checkpoints[self.current_checkpoint]

        if active_point.reached(self.quadcopter, self.controllers):
            self.current_checkpoint += 1

        return False


def controller(control_drones):
    quadcopter = Quadcopter(control_drones)

    flight_path = [
        RoutePoint(5.0, 0.0, 3.0, stable_steps=2),
        RoutePoint(0.0, 0.0, 3.0, stable_steps=2)
    ]

    navigation = NavigationSystem(quadcopter, flight_path)

    mission_complete = False

    while not mission_complete:
        mission_complete = navigation.tick()