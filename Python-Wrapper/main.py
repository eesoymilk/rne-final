import torch
from jetbotSim import Robot, Env, HumanAgent


def main() -> None:
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        env = Env()
        robot = Robot()
        agent = HumanAgent(env, robot)
        agent.run()
    except KeyboardInterrupt:
        print("\n[Exit]")


if __name__ == '__main__':
    main()
