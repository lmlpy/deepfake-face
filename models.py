import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=2,in_channels=3):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
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
        self.fc1 = nn.Linear(256 * 6 * 6, 512)
        self.dropout = nn.Dropout(0.5)  # Dropout层
        self.fc2 = nn.Linear(512, num_classes)
    def encoder(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #print(out.shape)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return out
    def forward(self, x):
        out = self.layer1(x)
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

class Facenet_(nn.Module):
    def __init__(self, num_classes=1):
        super(Facenet_, self).__init__()
        self.bn=nn.BatchNorm2d(3)
        self.cnn=CNN(num_classes)
    def interpolate(self, img, factor):
        #nearest
        return F.interpolate(F.interpolate(img, scale_factor=factor, mode='nearest',
                                           recompute_scale_factor=True), scale_factor=1/factor,
                             mode='nearest', recompute_scale_factor=True)
    def smooth_image(self,image_tensor, kernel_size=3, sigma=1.0):
        gaussian_blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        smoothed_image = gaussian_blur(image_tensor)

        return smoothed_image
    def encoder(self,x):
        NPR  = x - self.interpolate(x,0.5)
        NPR  = self.bn(NPR)
        feature = self.cnn.encoder(NPR)
        return feature
    def forward(self, x):
        NPR  = x - self.interpolate(x,0.5)
        NPR  = self.bn(NPR)
        out = self.cnn(NPR)
        return out

class Up_Down_Net(nn.Module):
    def __init__(self, factor=2):
        super(Up_Down_Net, self).__init__()

        self.factor = factor

        # 下采样卷积：这里使用普通的卷积，设置步幅来实现下采样
        self.downsample_conv = nn.Conv2d(
            in_channels=3,  # 输入是 RGB 图像
            out_channels=3,  # 输出通道数
            kernel_size=(1,2),  # 卷积核大小为 2,(1,2)模拟npr的interpolate函数
            stride=(1,2),  # 步幅为 2，表示每次下采样一倍
            padding=0,  # 无填充
            bias=False
        )
        # 上采样卷积：这里使用转置卷积 (ConvTranspose2d)，通过设置适当的 kernel_size 和 stride 来进行上采样
        self.upsample_conv = nn.ConvTranspose2d(
            in_channels=3,  # 假设输入是 RGB 图像，3 通道
            out_channels=3,  # 输出通道数
            kernel_size=(1,2),  # 卷积核大小为 2（一般转置卷积上采样倍数为 kernel_size）
            stride=(1,2),  # 步幅为 2，表示每次放大一倍
            padding=0,  # 无填充
            output_padding=0,  # 无输出填充
            bias=False
        )

        self.weight_init()
    def weight_init(self):
        down_value=torch.tensor([
            [[[1, 0],[0, 0]],
             [[0, 0],[0, 0]],
             [[0, 0],[0, 0]]],
            [[[0, 0],[0, 0]],
             [[1, 0],[0, 0]],
             [[0, 0],[0, 0]]],
            [[[0, 0],[0, 0]],
             [[0, 0],[0, 0]],
             [[1, 0],[0, 0]]]], dtype=torch.float32)
        up_value=torch.tensor([
            [[[1, 1],[1, 1]],
             [[0, 0],[0, 0]],
             [[0, 0],[0, 0]]],
            [[[0, 0],[0, 0]],
             [[1, 1],[1, 1]],
             [[0, 0],[0, 0]]],
            [[[0, 0],[0, 0]],
             [[0, 0],[0, 0]],
             [[1, 1],[1, 1]]]], dtype=torch.float32)
        down_value_nearst=torch.tensor([
            [[[1, 0]],
             [[0, 0]],
             [[0, 0]]],
            [[[0, 0]],
             [[1, 0]],
             [[0, 0]]],
            [[[0, 0]],
             [[0, 0]],
             [[1, 0]]]], dtype=torch.float32)
        up_value_nearst=torch.tensor([
            [[[1, 1]],
             [[0, 0]],
             [[0, 0]]],
            [[[0, 0]],
             [[1, 1]],
             [[0, 0]]],
            [[[0, 0]],
             [[0, 0]],
             [[1, 1]]]], dtype=torch.float32)
        self.downsample_conv.weight.data = down_value_nearst
        self.upsample_conv.weight.data = up_value_nearst
    def forward(self, img):
        img_downsampled = self.downsample_conv(img)# 下采样
        img_upsampled = self.upsample_conv(img_downsampled)# 上采样


        return img_upsampled
class Facenet(nn.Module):
    def __init__(self, num_classes=2):
        super(Facenet, self).__init__()
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
    def encoder(self,x):
        NPR  = x - self.interpolate(x, 0.5)
        out = self.layer1(NPR)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #print(out.shape)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return out
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
if __name__ == "__main__":
    input_tensor = torch.tensor([[[j*4+i+1 for i in range(4)] for j in range(4)] for k in range(3)],dtype=torch.float32).view((3,4,4))
    input_tensor = torch.rand((3,224,224))
    print(input_tensor)
    up_down_modle=Up_Down_Net(2)
    # 打印模型的具体参数
    for name, param in up_down_modle.named_parameters():
        if param.requires_grad:
            print(f"Parameter Name: {name}")
            print(f"Shape: {param.shape}")
            print(f"Values: {param.data}\n")
    down_tensor = up_down_modle.downsample_conv(input_tensor)
    print(down_tensor)
    output_tensor = up_down_modle.upsample_conv(down_tensor)
    print(output_tensor)

    factor = 0.5
    img = F.interpolate(F.interpolate(input_tensor, scale_factor=factor, mode='nearest',
                                recompute_scale_factor=False), scale_factor=1/factor,
                  mode='nearest', recompute_scale_factor=False)
    #print(F.interpolate(input_tensor, scale_factor=factor, mode='nearest',
                        #recompute_scale_factor=False))
    #print(img)
    #print((img-output_tensor))

    print(output_tensor==img)