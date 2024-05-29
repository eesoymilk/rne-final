import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))

import torch
from jetbotSim import Env, HumanAgent


def main() -> None:
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        env = Env()
        agent = HumanAgent(env)
        agent.run()
    except KeyboardInterrupt:
        print("\n[Exit]")


if __name__ == '__main__':
    main()
