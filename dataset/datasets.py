import numpy as np
import cv2
import sys
import torch

sys.path.append('..')

from torch.utils import data
from torch.utils.data import DataLoader


def flip(img, annotation):
    # parse
    img = np.fliplr(img).copy()
    h, w = img.shape[:2]

    x_min, y_min, x_max, y_max = annotation[0:4]
    landmark_x = annotation[4::2]
    landmark_y = annotation[4 + 1::2]

    bbox = np.array([w - x_max, y_min, w - x_min, y_max])
    for i in range(len(landmark_x)):
        landmark_x[i] = w - landmark_x[i]

    new_annotation = list()
    new_annotation.append(x_min)
    new_annotation.append(y_min)
    new_annotation.append(x_max)
    new_annotation.append(y_max)

    for i in range(len(landmark_x)):
        new_annotation.append(landmark_x[i])
        new_annotation.append(landmark_y[i])

    return img, new_annotation


def channel_shuffle(img, annotation):
    if (img.shape[2] == 3):
        ch_arr = [0, 1, 2]
        np.random.shuffle(ch_arr)
        img = img[..., ch_arr]
    return img, annotation


def random_noise(img, annotation, limit=[0, 0.2], p=0.5):
    if random.random() < p:
        H, W = img.shape[:2]
        noise = np.random.uniform(limit[0], limit[1], size=(H, W)) * 255

        img = img + noise[:, :, np.newaxis] * np.array([1, 1, 1])
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img, annotation


def random_brightness(img, annotation, brightness=0.3):
    alpha = 1 + np.random.uniform(-brightness, brightness)
    img = alpha * image
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, annotation


def random_contrast(img, annotation, contrast=0.3):
    coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
    alpha = 1.0 + np.random.uniform(-contrast, contrast)
    gray = img * coef
    gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
    img = alpha * img + gray
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, annotation


def random_saturation(img, annotation, saturation=0.5):
    coef = nd.array([[[0.299, 0.587, 0.114]]])
    alpha = np.random.uniform(-saturation, saturation)
    gray = img * coef
    gray = np.sum(gray, axis=2, keepdims=True)
    img = alpha * img + (1.0 - alpha) * gray
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, annotation


def random_hue(image, annotation, hue=0.5):
    h = int(np.random.uniform(-hue, hue) * 180)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + h) % 180
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image, annotation


def scale(img, annotation):
    f_xy = np.random.uniform(-0.4, 0.8)
    origin_h, origin_w = img.shape[:2]

    bbox = annotation[0:4]
    landmark_x = annotation[4::2]
    landmark_y = annotation[4 + 1::2]

    h, w = int(origin_h * f_xy), int(origin_w * f_xy)
    image = resize(img, (h, w), preserve_range=True, anti_aliasing=True, mode='constant').astype(np.uint8)

    new_annotation = list()
    for i in range(len(bbox)):
        bbox[i] = bbox[i] * f_xy
        new_annotation.append(bbox[i])

    for i in range(len(landmark_x)):
        landmark_x[i] = landmark_x[i] * f_xy
        landmark_y[i] = landmark_y[i] * f_xy
        new_annotation.append(landmark_x[i])
        new_annotation.append(landmark_y[i])

    return image, new_annotation


def rotate(img, annotation, alpha=30):
    bbox = annotation[0:4]
    landmark_x = annotation[4::2]
    landmark_y = annotation[4 + 1::2]
    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
    img_rotated_by_alpha = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))

    point_x = [bbox[0], bbox[2], bbox[0], bbox[2]]
    point_y = [bbox[1], bbox[3], bbox[3], bbox[1]]

    new_point_x = list()
    new_point_y = list()
    for (x, y) in zip(landmark_x, landmark_y):
        new_point_x.append(rot_mat[0][0] * x + rot_mat[0][1] * y + rot_mat[0][2])
        new_point_y.append(rot_mat[1][0] * x + rot_mat[1][1] * y + rot_mat[1][2])

    new_annotation = list()
    new_annotation.append(min(new_point_x))
    new_annotation.append(min(new_point_y))
    new_annotation.append(max(new_point_x))
    new_annotation.append(max(new_point_y))

    for (x, y) in zip(landmark_x, landmark_y):
        new_annotation.append(rot_mat[0][0] * x + rot_mat[0][1] * y + rot_mat[0][2])
        new_annotation.append(rot_mat[1][0] * x + rot_mat[1][1] * y + rot_mat[1][2])

    return img_rotated_by_alpha, new_annotation


def generate_FT(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fimg = np.log(np.abs(fshift) + 1)
    maxx = -1
    minn = 100000
    for i in range(len(fimg)):
        if maxx < max(fimg[i]):
            maxx = max(fimg[i])
        if minn > min(fimg[i]):
            minn = min(fimg[i])
    fimg = (fimg - minn + 1) / (maxx - minn + 1)
    return fimg


def draw_labelmap(img, pt, sigma=1, type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py

    # Check that any part of the gaussian is in-bounds
    ul = [int(int(pt[0]) - 3 * sigma), int(int(pt[1]) - 3 * sigma)]
    br = [int(int(pt[0]) + 3 * sigma + 1), int(int(pt[1]) + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img


class WLFWDatasets(data.Dataset):
    def __init__(self, file_list, transforms=None):
        self.line = None
        self.lm_number = 98
        self.img_size = 96
        self.ft_size = self.img_size // 2
        self.hm_size = self.img_size // 2
        self.transforms = transforms
        with open(file_list, 'r') as f:
            self.lines = f.readlines()

    def __getitem__(self, index):
        self.line = self.lines[index].strip()
        jpg_idx = self.line.find('png')
        line_data = [self.line[:jpg_idx + 3]]
        line_data.extend(self.line[jpg_idx + 4:].split())
        self.line = line_data

        self.img = cv2.imread(self.line[0])

        # generate ft
        # self.ft_img = generate_FT(self.img)
        # self.ft_img = cv2.resize(self.ft_img, (self.ft_size, self.ft_size))
        # self.ft_img = torch.from_numpy(self.ft_img).float()
        # self.ft_img = torch.unsqueeze(self.ft_img, 0)

        self.landmark = np.asarray(self.line[1:197], dtype=np.float32)

        # generate heatmap
        # self.heatmaps = np.zeros((self.lm_number, self.img_size, self.img_size))
        # for idx in range(self.lm_number):
        #     self.heatmaps[idx, :, :] = draw_labelmap(self.heatmaps[idx, :, :], (self.landmark[idx * 2] * self.img_size, self.landmark[idx * 2 + 1] * self.img_size))
        # self.heatmap = cv2.resize(self.heatmap, (self.hm_size, self.hm_size))
        # self.heatmap = (self.heatmap * 255).astype(np.uint8)
        # with open("heatmap.txt", "w") as f:
        #     for i in range(self.hm_size):
        #         str_ = ','.join(str(s) for s in self.heatmap[i, :])
        #         f.write(str_ + '\n')
        # cv2.imwrite('heatmap.jpg', self.heatmap)

        if self.transforms:
            self.img = self.transforms(self.img)
        return self.img, self.landmark

    def __len__(self):
        return len(self.lines)


if __name__ == '__main__':
    file_list = './data/test_data/list.txt'
    wlfwdataset = WLFWDatasets(file_list)
    dataloader = DataLoader(wlfwdataset, batch_size=256, shuffle=True, num_workers=0, drop_last=False)
    for img, landmark, attribute, euler_angle in dataloader:
        print("img shape", img.shape)
        print("landmark size", landmark.size())
        print("attrbute size", attribute)
        print("euler_angle", euler_angle.size())
