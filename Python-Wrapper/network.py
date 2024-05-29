import torch.nn as nn
import torch.nn.functional as F
import torch

def get_unet(n_classes=3, device="cpu"):
    net = torch.hub.load(
        'milesial/Pytorch-UNet',
        'unet_carvana',
        scale=0.5,
    )
    # chkpt = torch.hub.load_state_dict_from_url(
    #     "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth",
    #     map_location=device,
    # )
    # net.load_state_dict(chkpt)

    net.outc = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)
    return net

def predict(net, obs, device="cpu"):
    obs = torch.FloatTensor(obs / 255.).permute(2, 0, 1).unsqueeze(0).to(device)
    output = net(obs)
    mask = output.argmax(dim=1).cpu().numpy()
    return mask


if __name__ == "__main__":
    import cv2
    import numpy as np
    net = get_unet()
    net.load_state_dict(torch.load("unet_segmentation.pth", map_location="cpu"))
    img = cv2.imread("camera_test.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = predict(net, img)
    
    colors = [
        [200, 200, 255],
        [0, 0, 0],
        [255, 0, 0],
    ]
    print(mask.shape)
    img = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i in range(3):
        img[mask == i] = colors[i]
    cv2.imshow("image", cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)