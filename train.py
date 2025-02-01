import os
import shutil
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4)
        )
        self.maxpool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256 * 6 * 6, 512)
        self.dropout = nn.Dropout(0.5)  # Dropout层
        self.fc2 = nn.Linear(512, num_classes)

    def interpolate(self, img, factor):
        return F.interpolate(F.interpolate(img, scale_factor=factor, mode='nearest',
                                           recompute_scale_factor=True), scale_factor=1/factor,
                             mode='nearest', recompute_scale_factor=True)

    def forward(self, x):
        NPR  = x - self.interpolate(x, 0.5)
        out = self.layer1(NPR)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #print(out.shape)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)  # 应用Dropout
        out = F.relu(out)
        out = self.fc2(out)
        return out

if __name__=="__main__":
    # 数据目录
    data_dir = r"D:\third\src"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # 数据增强和归一化
    data_transforms = {
        'traindata': transforms.Compose([
            #transforms.Lambda(lambda img: interpolate(img)),
            transforms.CenterCrop(224),
            #transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valdata': transforms.Compose([
            #transforms.Lambda(lambda img: interpolate(img)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'testdata': transforms.Compose([
            #transforms.Lambda(lambda img: interpolate(img)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 加载数据集
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['traindata', 'valdata']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32,
                                 shuffle=True, num_workers=0)
                   for x in ['traindata', 'valdata']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['traindata', 'valdata']}
    class_names = image_datasets['traindata'].classes
    print(class_names)
    # 检查数据集大小
    print(f'Dataset sizes: {dataset_sizes}')
    print(f'Class names: {class_names}')

    # 实例化模型并移动到设备上
    model = SimpleCNN(num_classes=2).to(device)
    summary(model, (3, 224, 224))
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 训练模型
    num_epochs = 20
    best_acc=0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['traindata', 'valdata']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                # 将数据移动到设备上
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'traindata'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'traindata':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase!="traindata":
                if epoch_acc>best_acc:
                    best_acc=epoch_acc
                    #模型存储
                    torch.save({
                        'model_state_dict': model.state_dict(),  # 存储模型权重
                    }, f'./weights/facenet_model_{epoch_acc:.4f}.pth')
                    #torch.save(model.state_dict(), f'./weights/facenet_model_{epoch_acc:.4f}.pth')  # 保存模型状态
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    #模型存储
    torch.save({
        'model_state_dict': model.state_dict(),  # 存储模型权重
    }, f'./weights/facenet_model_last.pth')