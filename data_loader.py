from torch.utils.data import Dataset
from torchvision import transforms
import torch
import cv2
import numpy as np

def resize_image(image, width, height):
    top, bottom, left, right = (0, 0, 0, 0)
    # 获取图像尺寸
    h, w, _ = image.shape
    # 计算短边需要增加多上像素宽度使其与长边等长
    if h < w:
        dh = w - h
        top = dh // 2
        bottom = dh - top
    else:
        dw = h - w
        left = dw // 2
        right = dw - left
    # else:   #考虑相等的情况（感觉有点多余，其实等于0时不影响结果）
    #     pass
    # RGB颜色
    BLACK = [0, 0, 0]
    # 给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    # 调整图像大小并返回
    constant = cv2.resize(constant, (width, height))
    return constant

class FaceLandmarksDataset(Dataset):
    def __init__(self, txt_file):
        self.transform = transforms.Compose([transforms.ToTensor()])
        lines = []
        with open(txt_file) as read_file:
            for line in read_file:
                line = line.replace('\n', '')
                lines.append(line)
        self.landmarks_frame = lines

    def __len__(self):
        return len(self.landmarks_frame)

    def num_of_samples(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
           idx = idx.tolist()
        contents = self.landmarks_frame[idx].split(';')

        image_path = './dataset/'+contents[0]
        # print(image_path)
        img = cv2.imread(image_path)  # BGR order

        # img = cv2.resize(img, (64, 64))
        img = resize_image(img,96,96)
        img = (np.float32(img) /255.0 - 0.5) / 0.5

        landmarks_1 = np.array(contents[1])
        landmarks_2 = np.array(contents[2])

        landmarks_1 = torch.from_numpy(landmarks_1.astype('float32'))
        landmarks_2 = torch.from_numpy(landmarks_2.astype('float32'))

        # landmarks_1 = torch.nn.functional.one_hot(landmarks_1, 3)
        # landmarks_2 = torch.nn.functional.one_hot(landmarks_2, 3)
        # H, W C to C, H, W
        img = img.transpose((2, 0, 1))
        sample = {'image': torch.from_numpy(img),
                  'landmarks1': landmarks_1,
                  'landmarks2': landmarks_2,
                  }
        return sample
