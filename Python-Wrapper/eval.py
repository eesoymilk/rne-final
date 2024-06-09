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

        ver = input("version of model? ")
        chkpt = SCRIPT_DIR / "models" / f"ddqn_{ver}.chkpt"
        agent = Agent(
            Env(turn_speed=0.15 if input("This model was trained on 0.1 turn rate. Do you want to increase it for eval?\nThis is recommended for models with version values under 35. (y/n): ").lower() == "y" else 0.1),
            action_dim=6,
            checkpoint=chkpt,
            device=device,
            memory_size=750_000,
        )
        agent.eval(10)
    except KeyboardInterrupt:
        print("\n[Interrupted]")
    finally:
        print("[Exit]")


if __name__ == '__main__':
    main()
