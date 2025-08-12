import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import onnxruntime as ort
import os

# --- 1. 定义 LeNet-5 神经网络模型 ---
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return torch.log_softmax(x, dim=1)

# --- 2. 数据加载和预处理 (增加数据增强) ---
# 定义训练集的变换，增加随机旋转、平移、缩放和颜色抖动
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),       # 随机旋转，模拟拍摄角度偏差
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), # 随机平移和缩放
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # 模拟光照变化
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

# 定义测试集的变换，只做标准化，不增强
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# --- 3. 训练模型 ---
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')

# --- 4. 初始化模型和优化器 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet5().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 5. 训练模型 (可以根据需要调整 epoch 数量) ---
epochs = 20 # 增加训练周期，以更好地学习增强后的数据
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

# --- 6. 将 PyTorch 模型导出为 ONNX ---
dummy_input = torch.randn(1, 1, 28, 28, device=device)
onnx_model_path = "best.onnx" # 使用 'best.onnx' 以覆盖旧文件
torch.onnx.export(model,
                  dummy_input,
                  onnx_model_path,
                  verbose=False,
                  opset_version=11,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}})

print(f"PyTorch LeNet-5 model exported to {onnx_model_path}")

# --- 7. (可选) 使用 ONNX Runtime 验证导出的模型 ---
try:
    ort_session = ort.InferenceSession(onnx_model_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    print("\nONNX model loaded and verified with ONNX Runtime.")
except Exception as e:
    print(f"Error verifying ONNX model with ONNX Runtime: {e}")