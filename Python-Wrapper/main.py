import torch
from jetbotSim import Robot, Env, HumanAgent


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = Env()
    robot = Robot()
    agent = HumanAgent(env, robot)
    agent.run()


if __name__ == '__main__':
    main()
