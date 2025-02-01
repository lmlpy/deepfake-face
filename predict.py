import os
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import models

def load_model(model_path, num_classes=2):
    model = models.Facenet(num_classes)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()
    return model

def predict_image(model, image_path, transform):
    # 读取并转换图像
    image = Image.open(image_path)
    # 检查图像模式
    if image.mode != 'RGB':
        # 如果是灰度图，转换为RGB
        image = image.convert('RGB')
    image = transform(image).unsqueeze(0)  # 增加batch维度
    return image

if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = r"D:\third\src\weights\facenet_model_0.9519.pth"  # 模型路径

    # 加载模型
    model = load_model(model_path).to(device)

    # 定义图像转化
    data_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 测试图像路径
    test_root='./C2'
    test_image_paths = os.listdir(test_root)
    test_image_paths.sort()
    print(test_image_paths)
    # 进行预测
    res=[]
    for image_path in test_image_paths:
        image_tensor = predict_image(model, os.path.join(test_root,image_path), data_transform).to(device)
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
        label_pre=str(abs(predicted.item()-1))
        print(f'Image: {image_path}, Predicted class: {label_pre}')
        res.append(image_path[:-4] + ',' + label_pre)

        """# 显示图像
        plt.imshow(Image.open(os.path.join(test_root,image_path)))
        plt.title(f'Predicted: {label_pre}')
        plt.axis('off')
        plt.show()"""
    # 将预测结果保存到result_save_path
    result_save_path="./cla_pre.csv"
    with open(result_save_path, 'w') as f:
        f.write('\n'.join(res))