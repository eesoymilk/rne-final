from real_env import RealEnv
from real_agent import RealAgent, RuleBasedAgent
from pathlib import Path
import torch
import cv2
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import os
from datetime import datetime
import sys

colors = [
    [200, 200, 255],
    [0, 0, 0],
    [255, 0, 0],
]

# colors = {
#     (1, 0, 0): [200, 200, 255],
#     (0, 1, 0): [0, 0, 0],
#     (0, 0, 1): [255, 0, 0],
# }

dirs = {
    0: "forward",
    1: "left",
    2: "right",
    3: "backward",
    4: "s_left",
    5: "s_right"
}

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    write_image = True
    if len(sys.argv) > 1:
        write_image = False
    print(device)
    checkpoint = Path("ddqn_60_turbo.chkpt")
    unet = Path("gray_unet_84_84.pth")
    red_unet = Path("red_unet_84_84.pth")
    env = RealEnv(forward_speed=0.2, turn_speed=0.12)
    try:
        picture_path = f"{datetime.now():%m%d%H%M}"
        os.mkdir(picture_path)
    except:
        pass
    # test_env()
    # agent = RealAgent(
    agent = RuleBasedAgent(
        checkpoint=checkpoint,
        action_dim=6,
        device=device,
        unet=unet,
        red_unet = red_unet
    )
    # test_agent()

    try:
        obs = env.reset()
        i = 0
        # _one = torch.tensor(1.).to(device)
        # _zero = torch.tensor(0.).to(device)
        print("starting")
        while True:
            # action = agent.get_action(obs)
            # action = random.randint(0, 3)
            gray_mask, image = agent.segment_test(obs)
            rg_mask = agent.segment_test_rg(obs)

            '''
            obs_tensor = F.one_hot(gray_mask, num_classes=3).float().permute(0, 3, 1, 2) # 1, 3, 84, 84
            rg_obs_tensor = F.one_hot(rg_mask, num_classes=3).float().permute(0, 3, 1, 2) # 1, 3, 84, 84

            # track
            # print(type(obs_tensor[:, 1, :, :]))
            print("merging track to minimum")
            obs_tensor[:, 1, :, :] = obs_tensor[:, 1, :, :].where((obs_tensor[:, 1, :, :] == 1) & (rg_obs_tensor[:, 1, :, :] == 1), _zero)
            # obstacle
            print("merging obstacle to minimum")
            obs_tensor[:, 2, :, :] = obs_tensor[:, 2, :, :].where((obs_tensor[:, 2, :, :] == 1) & (rg_obs_tensor[:, 2, :, :] == 1), _zero)
            # expand background
            print("expanding background")
            obs_tensor[:, 0, :, :] = obs_tensor[:, 0, :, :].where((obs_tensor[:, 1, :, :] == 0) & (obs_tensor[:, 2, :, :] == 0), _one)
            print("Merging done")
            '''
            mask = torch.zeros_like(gray_mask, device=device)
            mask[(gray_mask == 1) & (rg_mask == 1)] = 1
            # mask[(gray_mask == 2) & (rg_mask == 2)] = 2
            mask[(gray_mask == 2) & (mask != 1)] = 2

            obs_tensor = F.one_hot(mask, num_classes=3).float().permute(0, 3, 1, 2) # 1, 3, 84, 84

            action_values = agent.net(obs_tensor, model="online")
            action = torch.argmax(action_values, axis=1).item()
            # print(action, action_values)
            if write_image:
                # mask = obs_tensor.permute(0, 2, 3, 1).cpu().numpy()
                mask = mask.cpu().numpy()
                seg_img = np.zeros((1, 84, 84, 3), dtype=np.uint8)
                # mask_ = torch.argmax(mask, axis=1).cpu().numpy()
                ####### NOT WORKING #######
                # for j in range(3):
                #     seg_img[:, :, :] = colors[j] if mask[:, :, :][j] == 1 else seg_img[:, :, :]
                ####### NOT WORKING #######

                for j in range(3):
                    seg_img[mask == j] = colors[j]
                plt.subplot(1, 2, 1)
                plt.imshow(obs[..., [2, 1, 0]])
                plt.title("Original Image")
                # Show the segmented image
                plt.subplot(1, 2, 2)
                plt.imshow(seg_img[0])
                plt.title(f"Segmented Image_action{dirs[action]}")

                # Save the result
                plt.savefig(f"{picture_path}/{i}_{dirs[action]}.png")
                # cv2.imwrite(f"real_env/{i}.png", obs)
            obs, _, _ = env.step(action)
            i += 1
    except KeyboardInterrupt:
        print("quitting...")
    finally:
        env.close()

def test_agent():
    global agent
    obs = cv2.imread("plays/05300431/00005.png")
    obs = cv2.resize(obs, (84, 84))
    print(agent.get_action(obs))
    
    segment = agent.segment(obs)
    print(segment.shape)
    torch.set_printoptions(threshold=100_000)
    open("test.txt", "w").write(str(segment))

def test_env():
    global env
    i = 0
    try:
        while i < 100:
            if i < 10:
                obs, _, _ = env.step(0)
            elif i < 20:
                obs, _, _ = env.step(1)
            elif i < 30:
                obs, _, _ = env.step(2)
            elif i < 40:
                obs, _, _ = env.step(3)
            else:
                obs, _, _ = env.step(4)
            cv2.imwrite(f"real_env/obs_{i}.png", obs)
            time.sleep(0.1)
            print(i)
            i += 1
    except KeyboardInterrupt:
        pass
    env.close()
    


if __name__ == "__main__":
    main()