import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))

import torch
from jetbot_sim.environment import Env
from ddqn.agent import Agent


def main() -> None:
    try:
        print("[Start]")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[Device]:", device)

        chkpt = SCRIPT_DIR / "models" / "ddqn_16.chkpt"
        agent = Agent(
            Env(turn_speed=0.15),
            action_dim=4,
            checkpoint=chkpt,
            device=device,
        )
        agent.eval(10)
    except KeyboardInterrupt:
        print("\n[Interrupted]")
    finally:
        print("[Exit]")


if __name__ == '__main__':
    main()
