import torch
import torchvision
import torchvision.transforms as transforms
from network import Net, Vgg16_net, AlexNet
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt

batch_size = 16
class_num = 10
epoches = 20

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

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net().to(device)
# net = AlexNet().to(device)
# net = Vgg16_net().to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(),
                lr=0.0001,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=5e-7,
                amsgrad=False)
#定义两个数组
Loss_list = []
Accuracy_list = []
net = Net()
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print('Start Training')

for epoch in range(epoches):
    running_loss = 0.0  # 定义一个变量方便我们对loss进行输出
    since = time.time()
    correct = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum()  # 更新正确分类的图片的数量
        running_loss += loss.item()

        if i == int(50000 / batch_size) - 1:
            print(i)
            Loss_list.append(running_loss / 2000)
            Accuracy_list.append(100 * correct.cpu() / 2000)
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))  # 然后再除以2000，就得到这两千次的平均损失值
            running_loss = 0.0  # 这一个2000次结束后，就把running_loss归零，下一个2000次继续使用
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

print('Finished Training')

#我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上
x1 = range(0, epoches)
x2 = range(0, epoches)
y1 = Accuracy_list
y2 = Loss_list
print(type(y1))
plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('Test accuracy vs. epoches')
plt.ylabel('Test accuracy')
plt.subplot(2, 1, 2)
plt.plot(x2, y2, '.-')
plt.xlabel('Test loss vs. epoches')
plt.ylabel('Test loss')
plt.show()
# torch.save(net, 'model.pt')

correct = 0
total = 0
for data in testloader:  # 循环每一个batch
    images, labels = data
    outputs = net(Variable(images))  # 输入网络进行测试
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)  # 更新测试图片的数量
    correct += (predicted == labels).sum()  # 更新正确分类的图片的数量

print('Accuracy of the network on the 10000 test images: %.2f %%' % (
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
    print('Accuracy of %5s : %.2f %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))