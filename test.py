import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
# from multiprocessing import set_start_method


batch_size = 16
class_num = 10
epoches = 20

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)



net = torch.load('model.pt')

correct = 0
total = 0
for data in testloader:  # 循环每一个batch
    images, labels = data
    images = Variable(images).to(device)
    labels = Variable(labels).to(device)
    outputs = net(images)  # 输入网络进行测试
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)  # 更新测试图片的数量
    correct += (predicted == labels).sum()  # 更新正确分类的图片的数量

print('Accuracy of the network on the 10000 test images: %.2f %%' % (
        100 * correct / total))  # 打印结果


class_correct = list(0. for i in range(class_num))  # 定义一个存储每类中测试正确的个数的 列表，初始化为0
class_total = list(0. for i in range(class_num))  # 定义一个存储每类中测试总数的个数的 列表，初始化为0
for data in testloader:  # 以一个batch为单位进行循环
    images, labels = data
    images = Variable(images).to(device)
    labels = Variable(labels).to(device)
    outputs = net(images)  # 输入网络进行测试
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(batch_size):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

for i in range(class_num):
    print('Accuracy of %5s : %.2f %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
