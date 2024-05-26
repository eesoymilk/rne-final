import torch
from jetbotSim import Robot, Env, Agent


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = Env()
    robot = Robot()
    agent = Agent(env, robot, device=device)
    agent.run()


if __name__ == '__main__':
    main()
