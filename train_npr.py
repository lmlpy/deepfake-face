import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
import models

if __name__=="__main__":
    # 数据目录
    data_dir = r"D:\third\src"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # 数据增强和归一化
    data_transforms = {
        'traindata': transforms.Compose([
            #transforms.Lambda(lambda img: interpolate(img)),
            transforms.CenterCrop(256),
            #transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valdata': transforms.Compose([
            #transforms.Lambda(lambda img: interpolate(img)),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'testdata': transforms.Compose([
            #transforms.Lambda(lambda img: interpolate(img)),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 加载数据集
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x if x=="traindata" else "DALLE"),
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
    model = models.Facenet(num_classes=2)
    model_path=r"D:\third\src\weights\facenet_model_0.9519.pth"
    #model.load_state_dict(torch.load(model_path,weights_only=False)['model_state_dict'])
    model.eval()
    model.to(device)
    summary(model, (3, 256, 256))
    """# 冻结除第一层外的所有参数
    for name, param in model.named_parameters():
        print(name)
        if name == 'fc1.weight' and name == 'fc1.bias':
            param.requires_grad = False"""
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)


    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

    # 训练模型
    num_epochs = 50
    best_acc=0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['valdata','traindata']:
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
                if epoch_acc>best_acc and epoch_acc>0.9:
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