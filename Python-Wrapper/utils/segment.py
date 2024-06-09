import cv2
import numpy as np
import numpy.typing as npt

def segment(img: npt.NDArray, one_hot: bool = True) -> npt.NDArray:
    '''Segmentation by pure color detection.
    Applicable to the simulation environment.'''
    blue = img[:, :, 0]
    green = img[:, :, 1]
    red = img[:, :, 2]
    # Create a result array with the same height and width, initialized to 1
    result = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)

    # Apply the conditions to assign the segmentation labels
    result[(red - 5 > green) & (red - 5 > blue)] = 2
    result[blue > 200] = 0
    random_point = 250
    result[0:20, :] = 0
    for i in range(3):
        for _ in range(random_point):
            x = np.random.randint(0, result.shape[0])
            y = np.random.randint(0, result.shape[1])
            result[x, y] = i

    if one_hot:
        result = np.eye(3, dtype=np.bool_)[result].transpose(2, 0, 1)

    # print(result.shape)
    return result


def decode_segmented(mask):
    '''Decode the segmented masks to an image. Just for visualization.'''
    colors = [
        [255, 200, 200],  # BGR
        [0, 0, 0],
        [0, 0, 255],
    ]
    img = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        img[mask == i] = color
    return img


def main() -> None:
    img = cv2.imread("input.png")
    img = cv2.resize(img, (84, 84))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = segment(img, one_hot=False)
    img = decode_segmented(mask)
    print(img.shape)
    # cv2.imshow("image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)
    cv2.imwrite("sim_segment.png", img) #cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()
