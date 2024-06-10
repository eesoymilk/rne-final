import torch.nn as nn
import torch.nn.functional as F
import torch
import time

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
    net.eval().to(device)
    return net

def predict(net, obs, device="cpu"):
    start = time.time()
    obs = torch.FloatTensor(obs / 255.).permute(2, 0, 1).unsqueeze(0).to(device)
    output = net(obs)
    mask = output.argmax(dim=1).cpu().numpy()
    end = time.time()
    print("time:", end-start)
    return mask

def get_unet_trt(file="net_trt.pth"):
    from torch2trt import TRTModule
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(file))
    print("load successfully")
    return model_trt

if __name__ == "__main__":
    import cv2
    import numpy as np
    import sys
    # device = 'cpu' if len(sys.argv) < 2 else sys.argv[1]
    device = 'cuda'
    # net = get_unet(device=device)
    net = get_unet_trt()
    img = cv2.imread("/home/jetbot/jetbot/notebooks/collision_avoidance/dataset/blocked/5fbd3630-1c38-11ef-b8dc-4602bd533a20.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = predict(net, img, device=device)
    
    colors = [
        [200, 200, 255],
        [0, 0, 0],
        [255, 0, 0],
    ]
    print(mask.shape)
    img = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i in range(3):
        img[mask == i] = colors[i]
    cv2.imwrite("./test_img/test_img0.jpg", cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)

#     data = torch.zeros((1, 3, 224, 224)).cuda().half()

#     model_trt = torch2trt(net, [data], fp16_mode=True)
#     torch.save(model_trt.state_dict(), 'best_model_trt.pth')

#     mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
#     std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()

#     normalize = torchvision.transforms.Normalize(mean, std)

# def preprocess(image):
#     image = PIL.Image.fromarray(image)
#     image = transforms.functional.to_tensor(image).to('cuda').half()
#     image.sub_(mean[:, None, None]).div_(std[:, None, None])
#     return image[None, ...]
