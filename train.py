from data_loader import FaceLandmarksDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable

num_epochs = 8
batch_size = 256
learning_rate = 0.001

ds = FaceLandmarksDataset("./dataset/train.txt")
dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)

def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x

# 两层卷积
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=3,padding=1),
            # nn.BatchNorm2d(8),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            torch.nn.Conv2d(8, 16, kernel_size=3,padding=1),
            # nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3,padding=1),
            # nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )
        self.fc1 = nn.Sequential(
			nn.Linear(32*12*12, 128),
			nn.Linear(128, 3))

        self.fc2 = nn.Sequential(
            nn.Linear(32*12*12, 128),
            nn.Linear(128, 3))

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        # print(out.shape)
        out = out.view(out.size(0), -1)  # reshape
        out1 = self.fc1(out)
        out2 = self.fc2(out)
        return out1,out2

cnn = CNN()
if torch.cuda.is_available():
    cnn = cnn.cuda()

loss_func = nn.CrossEntropyLoss()
# loss_func2 = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    train_loss = 0
    train_acc = 0
    for i, data in enumerate(dataloader):
        images = get_variable(data['image'])
        labels1 = get_variable(data['landmarks1'])
        labels2 = get_variable(data['landmarks2'])

        outputs1,outputs2 = cnn(images)
        loss1 = loss_func(outputs1, labels1.long())#torch.LongTensor
        loss2 = loss_func(outputs2, labels2.long())

        # print(loss1.item(),loss2.item())
        loss = loss1 + loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred1 = outputs1.argmax(dim=1)
        pred2 = outputs2.argmax(dim=1)

        train_acc += (pred1 == labels1).sum().item() / images.size(0)
        train_acc += (pred2 == labels2).sum().item() / images.size(0)

        loss += (loss1+loss2)

    train_loss = loss / len(dataloader)
    train_acc = train_acc / len(dataloader) / 2
    # train_losses.append(train_loss)
    # train_accs.append(train_accs)
    print('loss:{},acc:{}'.format(train_loss,train_acc))
torch.save(cnn.state_dict(), './models/model.pkl')