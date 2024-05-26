from jetbotSim import Robot, Env, Agent
import numpy as np
import cv2
import matplotlib.pyplot as plt


# def step(action):
#     global robot
#     if action == 0:
#         robot.set_motor(0.5, 0.5)
#     elif action == 1:
#         robot.set_motor(0.2, 0.0)
#     elif action == 2:
#         robot.set_motor(0.0, 0.2)


# def execute(obs):
#     # Visualize
#     global frames
#     frames += 1
#     img = obs["img"]
#     # cv2.imwrite(f'test_img.png', img)
#     reward = obs['reward']
#     done = obs['done']
#     if frames < 100:
#         step(0)
#     else:
#         frames = 0
#         robot.reset()
#     print(f'\rframes:{frames}, reward:{reward}, done:{done} ', end="")


# frames = 0
# robot = Robot()
# env = Env()
# env.run(execute)


def main() -> None:
    env = Env()
    robot = Robot()
    agent = Agent(env, robot)
    agent.run()


if __name__ == '__main__':
    main()
