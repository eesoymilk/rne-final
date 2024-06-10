import PIL.Image
import torch
import cv2
import numpy as np
import PIL
from ddqn import JetbotDDQN
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch2trt import TRTModule
from network import get_unet

class RealAgent:
    def __init__(self, checkpoint, 
                 action_dim,
                 device,
                 unet,
                 red_unet):
        self.net = JetbotDDQN(action_dim)
        self.device = device
        self.processed_dim = (84, 84)
        # self.unet = TRTModule()
        # print("loading unet...")
        # self.unet.load_state_dict(torch.load(unet))
        self.unet = get_unet(device=device)
        self.unet.load_state_dict(torch.load(unet))

        self.red_unet = get_unet(device=device)
        self.red_unet.load_state_dict(torch.load(red_unet))
        print("unet loaded successfully")
        if checkpoint is not None:
            self.load(checkpoint)
        self.net.to(device)
    
    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        print(f"Loading model at {load_path}...")

        ckp: dict = torch.load(load_path, map_location=self.device)
        exploration_rate = ckp.get("exploration_rate")
        cnns = ckp.get("cnns")
        val_stream = ckp.get("value_stream")
        adv_stream = ckp.get("advantage_stream")
        self.net.cnns.load_state_dict(cnns)
        self.net.val_stream.load_state_dict(val_stream)
        self.net.adv_stream.load_state_dict(adv_stream)
        self.net.sync()
        self.exploration_rate = exploration_rate

        print(
            f"Model loaded successfully from {load_path} with exploration rate {exploration_rate}"
        )

    def image_preprocess_rg(self, image):
        # gray then concat
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        red = image[:, :, [2]]
        image = np.concatenate((red, np.repeat(gray, 2).reshape(*gray.shape, 2)), axis=2)
        image = cv2.resize(image, self.processed_dim)
        return image
    
    def image_preprocess(self, image):
        # gray then concat
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.repeat(gray, 3).reshape(*gray.shape, 3)
        gray = cv2.resize(gray, self.processed_dim)
        return gray

    def segment_postprocess(self, mask: torch.Tensor):
        # fucking softmax
        # mask[:, 0, 0:25, :] = 1
        # mask[:, 1:, 0:25, :] = 0
        
        # mask[:, -30:, :10] = 0 # background
        # mask[:, -30:, -10:] = 0
        # mask = mask.cpu()
        # mask = TF.to_pil_image(mask)
        # mask = TF.resize(mask, (84, 60), PIL.Image.NEAREST)
        # mask = TF.pad(mask, (12, 0))
        # mask = TF.pil_to_tensor(mask)
        # mask = mask.cuda()

        return mask

    def segment(self, image):
        # THIS IS CURRENTLY NOT USED, DON'T EDIT IT
        image = self.image_preprocess(image)
        obs = torch.FloatTensor(image / 255.).permute(2, 0, 1).unsqueeze(0).cuda() # 1, 3, 84, 84
        # mask = self.unet(obs).argmax(dim=1) # 1, 84, 84
        mask = F.softmax(self.unet(obs), dim=1) # 1, 84, 84
        mask = self.segment_postprocess(mask)
        
        # mask = F.one_hot(mask, num_classes=3).float().permute(0, 3, 1, 2) # 1, 3, 84, 84
        return mask

    def segment_test(self, image):
        image = self.image_preprocess(image)
        obs = torch.FloatTensor(image / 255.).permute(2, 0, 1).unsqueeze(0).cuda() # 1, 3, 84, 84
        mask = self.unet(obs).argmax(dim=1) # 1, 84, 84
        # mask = F.softmax(self.unet(obs), dim=1) # 1, 3, 84, 84
        # mask = self.segment_postprocess(mask)
        return mask, image
    
    def segment_test_rg(self, image):
        image = self.image_preprocess_rg(image)
        obs = torch.FloatTensor(image / 255.).permute(2, 0, 1).unsqueeze(0).cuda() # 1, 3, 84, 84
        mask = self.red_unet(obs).argmax(dim=1) # 1, 84, 84
        return mask

    def get_action(self, obs):
        obs_tensor = self.segment(obs)
        action_values = self.net(obs_tensor, model="online")
        action_idx = torch.argmax(action_values, axis=1).item()
        return action_idx


class RuleBasedAgent(RealAgent):
    def __init__(self, device, unet, red_unet, **kwargs):
        self.device = device
        self.processed_dim = (84, 84)
        self.unet = get_unet(device=device)
        self.unet.load_state_dict(torch.load(unet))

        self.red_unet = get_unet(device=device)
        self.red_unet.load_state_dict(torch.load(red_unet))
        print("unet loaded successfully")
        
    def net(self, obs_tensor, model="online"):
        """
        obs_tensor: torch.Tensor with shape (1, 3, 84, 84)
        """
        left_track = obs_tensor[:, 1, :, :42].sum()
        right_track = obs_tensor[:, 1, :, 42:].sum()
        left_low_obstacle = obs_tensor[:, 2, 42:, :42].sum()
        right_low_obstacle = obs_tensor[:, 2, 42:, 42:].sum()

        left_bottom_obstacle = obs_tensor[:, 2, -25:, :10].sum()
        right_bottom_obstacle = obs_tensor[:, 2, -25:, -10:].sum()
        center_track = obs_tensor[:, 1, -25:, 20:64].sum()
        
        action = torch.zeros(1, 6, dtype=torch.int)
        action_idx = 0
        if left_track + right_track != 0:
            print(left_track, right_track, left_low_obstacle, right_low_obstacle)
            if left_low_obstacle + right_low_obstacle > 700:
                action_idx = 3
            elif left_low_obstacle > 200: # 
                action_idx = 2
            elif right_low_obstacle > 200: #
                action_idx = 1
            elif left_bottom_obstacle > 30 and right_bottom_obstacle > 30 and center_track > 880:
                action_idx = 0
            elif left_bottom_obstacle > 60: # left corner has obstacle => sharp right
                action_idx = 5
            elif right_bottom_obstacle > 60: # right corner has obstacle => sharp left
                action_idx = 4
            elif left_track / right_track > 1.2: # left track more => left
                action_idx = 1 
            elif right_track / left_track > 1.2: # right track more => right
                action_idx = 2
        action[0, action_idx] = 1 
        return action