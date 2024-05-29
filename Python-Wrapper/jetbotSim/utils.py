import cv2
import numpy as np

def segment(img, one_hot=True):
    '''Segmentation by pure color detection.
    Applicable to the simulation environment.'''
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    result = np.zeros((h, w), dtype=int)
    for i in range(h):
        for j in range(w):
            red = img[i][j][0] - 5
            if red > img[i][j][1] and red > img[i][j][2]:
                result[i][j] = 2
            elif img[i][j][2] > 200:
                result[i][j] = 0
            else:
                result[i][j] = 1
    if one_hot:
        result = np.eye(3)[result]
    return result

colors = [
    [200, 200, 255],
    [0, 0, 0],
    [255, 0, 0],
]

def decode_mask(mask):
    '''Decode the mask to an image. Just for visualization.'''
    # mask = mask.argmax(dim=1)
    # mask = mask.cpu().numpy()
    img = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i in range(3):
        img[mask == i] = colors[i]
    return img 


if __name__ == "__main__":
    img = cv2.imread("input.png")
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = segment(img, one_hot=False)
    img = decode_mask(mask)
    print(img.shape)
    # cv2.imshow("image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)
    cv2.imwrite("sim_segment.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))