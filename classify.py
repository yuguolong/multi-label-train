import numpy as np
import argparse
import imutils
import pickle
import cv2
from data_loader import FaceLandmarksDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms

tran=transforms.Compose([
    transforms.ToTensor(),
])

# class CNN(torch.nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Sequential(
#             torch.nn.Conv2d(3, 8, kernel_size=3,padding=1),
#             # nn.BatchNorm2d(8),
#             torch.nn.MaxPool2d(2),
#             torch.nn.ReLU(),
#         )
#         self.conv2 = nn.Sequential(
#             torch.nn.Conv2d(8, 16, kernel_size=3,padding=1),
#             # nn.BatchNorm2d(32),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(2),
#         )
#         self.conv3 = nn.Sequential(
#             torch.nn.Conv2d(16, 32, kernel_size=3,padding=1),
#             # nn.BatchNorm2d(32),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(2),
#         )
#         self.fc1 = nn.Sequential(
# 			nn.Linear(32*12*12, 128),
# 			nn.Linear(128, 3))
#
#         self.fc2 = nn.Sequential(
#             nn.Linear(32*12*12, 128),
#             nn.Linear(128, 3))
#
#     def forward(self,x):
#         out = self.conv1(x)
#         out = self.conv2(out)
#         out = self.conv3(out)
#         # print(out.shape)
#         out = out.view(out.size(0), -1)  # reshape
#         out1 = self.fc1(out)
#         out2 = self.fc2(out)
#         return out1,out2
#
# model = CNN()

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_0_1 = nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=3, padding=1),
            # nn.BatchNorm2d(8),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
        )
        self.conv_0_2 = nn.Sequential(
            torch.nn.Conv2d(8, 16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.conv_0_3 = nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 12 * 12, 128),
            nn.Linear(128, 3),
            nn.Softmax())

        self.conv_1_1 = nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=3, padding=1),
            # nn.BatchNorm2d(8),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
        )
        self.conv_1_2 = nn.Sequential(
            torch.nn.Conv2d(8, 16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.conv_1_3 = nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(32 * 12 * 12, 128),
            nn.Linear(128, 3),
            nn.Softmax())

    def forward(self, x):
        out1 = self.conv_0_1(x)
        out1 = self.conv_0_2(out1)
        out1 = self.conv_0_3(out1)
        out1 = out1.view(out1.size(0), -1)  # reshape
        out1 = self.fc1(out1)

        out2 = self.conv_1_1(x)
        out2 = self.conv_1_2(out2)
        out2 = self.conv_1_3(out2)
        out2 = out2.view(out2.size(0), -1)  # reshape
        out2 = self.fc2(out2)
        return out1, out2


model = CNN()
model.load_state_dict(torch.load('./models/model.pkl'))

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str,default='models.h5',
	help="path to trained model model")
ap.add_argument("-l", "--labelbin", type=str,default='modles',
	help="path to label binarizer")
ap.add_argument("-i", "--image", type=str,default='examples/example_07.jpg',#example_0
	help="path to input image")
args = vars(ap.parse_args())

path = args["image"]
# load the image
image = cv2.imread(path)
output = imutils.resize(image, width=400)

# pre-process the image for classification
image = cv2.resize(image, (96, 96))
image = np.expand_dims(image, axis=0)
image = (np.float32(image) / 255.0 - 0.5) / 0.5
image = image.transpose((0, 3, 1, 2))
image = torch.Tensor(image)
# print(image.shape)

# load the trained convolutional neural network and the multi-label
# binarizer
mlb = pickle.loads(open(args["labelbin"], "rb").read())

# classify the input image then find the indexes of the two class
# labels with the *largest* probability
print("[INFO] classifying image...")
proba1,proba2 = model(image)
print(proba1,proba2)

predict1 = torch.argmax(proba1, axis=1)
predict2 = torch.argmax(proba2, axis=1)
# print(*np.max(pre_y,axis=2)[0])
# print(pre_y)

label1 = {'black': 0, 'blue': 1, 'red': 2}
label2 = {'jeans': 0, 'dress': 1, 'shirt': 2}


def get_key2(dct, value):
    return [k for (k, v) in dct.items() if v == value]


a = get_key2(label1, predict1)
b = get_key2(label2, predict2)

cv2.putText(output, '{}'.format(*a), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)
cv2.putText(output, '{}'.format(*b), (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)

cv2.imshow("Image1", output)
cv2.waitKey(0)


