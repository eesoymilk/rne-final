import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))

from datetime import datetime
import torch
from jetbotSim import Env, Agent
from jetbotSim.ddqn import JetbotDDQN


def main() -> None:
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        save_dir = (
            SCRIPT_DIR / "checkpoints" / f"{datetime.now():%Y%m%d-%H%M%S}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        env = Env()
        agent = Agent(
            env,
            obs_dim=(84, 84),
            action_dim=4,
            save_dir=save_dir,
            device=device,
        )
        agent.train()
    except KeyboardInterrupt:
        print("\n[Exit]")


if __name__ == '__main__':
    net = JetbotDDQN(4).float()
    out = net(torch.randn(1, 3, 84, 84), "online")
    # main()
