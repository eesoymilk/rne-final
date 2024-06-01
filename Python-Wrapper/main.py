import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))

import torch
from datetime import datetime
from jetbot_sim.environment import Env
from ddqn.agent import Agent


def main() -> None:
    try:
        print("[Start]")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[Device]:", device)

        save_dir = SCRIPT_DIR / "checkpoints" / f"{datetime.now():%m%d%H%M}"
        chkpt = SCRIPT_DIR / "models" / "ddqn_1070k.chkpt"
        save_dir.mkdir(parents=True, exist_ok=True)

        agent = Agent(
            Env(),
            action_dim=4,
            save_dir=save_dir,
            checkpoint=chkpt,
            device=device,
        )
        agent.exploration_rate = 0.95
        agent.train()
    except KeyboardInterrupt:
        agent.save()
        print("\n[Interrupted]")
    finally:
        print("[Exit]")


if __name__ == '__main__':
    main()
