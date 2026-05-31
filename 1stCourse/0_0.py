import numpy as np


def controller(control_drones):
    first_point = [1, 0, 2]
    second_point = [2, 0, 3]
    third_point = [3, 0, 4]

    control_drones(np.array([first_point]))

    for _ in range(3):
        control_drones(np.array([second_point]))

    control_drones(np.array([third_point]))

    control_drones(np.array([second_point]))
    control_drones(np.array([first_point]))

    while True:
        control_drones(np.array([first_point]))