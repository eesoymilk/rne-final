import sys

import torch
import torch.nn as nn
import numpy as np
import PIL
import matplotlib.pyplot as plt


def main() -> None:
    net: nn.Module = torch.hub.load(
        'milesial/Pytorch-UNet',
        'unet_carvana',
        # pretrained=True,
        scale=0.5,
    )
    chekpoint = torch.hub.load_state_dict_from_url(
        "https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale0.5_epoch2.pth",
        map_location='cpu',
    )
    net.load_state_dict(chekpoint)

    img = np.array(PIL.Image.open('car.png'))
    out = net(torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float() / 255)
    print(out.shape)
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].set_title('Input')
    ax[2].set_title('Output 1')
    ax[1].set_title('Output 0')
    ax[0].axis('off')
    ax[2].axis('off')
    ax[1].axis('off')
    ax[0].imshow(img)

    mask = out[0].detach().numpy()
    ax[2].imshow(img)
    ax[2].imshow(mask[0], alpha=0.5, cmap='Reds')
    ax[1].imshow(img)
    ax[1].imshow(mask[1], alpha=0.5, cmap='Greens')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
