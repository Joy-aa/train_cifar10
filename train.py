import torch
import torchvision
import torchvision.transforms as transforms
from network import Net
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time

batch_size = 4
class_num = 10
epoches = 10

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = Net()
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print('Start Training')

for epoch in range(epoches):
    running_loss = 0.0  # 定义一个变量方便我们对loss进行输出
    since = time.time()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print(i)
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))  # 然后再除以2000，就得到这两千次的平均损失值
            running_loss = 0.0  # 这一个2000次结束后，就把running_loss归零，下一个2000次继续使用
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

print('Finished Training')

correct = 0
total = 0
for data in testloader:  # 循环每一个batch
    images, labels = data
    outputs = net(Variable(images))  # 输入网络进行测试
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)  # 更新测试图片的数量
    correct += (predicted == labels).sum()  # 更新正确分类的图片的数量

print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))  # 打印结果


class_correct = list(0. for i in range(class_num))  # 定义一个存储每类中测试正确的个数的 列表，初始化为0
class_total = list(0. for i in range(class_num))  # 定义一个存储每类中测试总数的个数的 列表，初始化为0
for data in testloader:  # 以一个batch为单位进行循环
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(batch_size):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

for i in range(class_num):
    print('Accuracy of %5s : %4d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))