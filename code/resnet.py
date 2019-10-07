import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from torchvision.models.resnet import resnet50
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='My Alex Training')
parser.add_argument('--outf', default='./resnet_model', help='folder to save the model')
args = parser.parse_args()

BATCH_SIZE = 100
EPOCH = 90
LRs = [0.01, 0.001, 0.0005]
# LRs = [0.0001]

transform_train = transforms.Compose([
    transforms.RandomCrop(32, 4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2, )

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = resnet50(pretrained=True).to(device)
# net.load_state_dict(torch.load('model/alexnet_100.pth'))

criterion = nn.CrossEntropyLoss()
optimizers = [optim.SGD(net.parameters(), lr=i, momentum=0.9, weight_decay=5e-4) for i in LRs]

if __name__ == "__main__":
    print("Start Training, fine tuned pretrained Resnet18!")
    best_acc = 60
    writer = SummaryWriter(comment='resnet18')
    net_input = torch.rand(BATCH_SIZE, 3, 32, 32).to(device)
    # writer.add_graph(net, (net_input,))
    with open("resnet_acc.txt", "w") as f:
        with open("resnet_log.txt", "w") as f2:
            for epoch in range(EPOCH):
                print("\nEpoch: %d" % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                optimizer = optimizers[epoch//30]
                for i, data in enumerate(train_loader, 0):
                    length = len(train_loader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss:%.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('[epoch:%d, iter:%d] Loss:%.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

                train_loss = sum_loss / total
                train_acc = 100. * correct / total
                writer.add_scalar('Loss', train_loss, epoch)

                print("Waiting Test")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in test_loader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)

                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    acc = 100. * correct / total
                    print("测试分类准确率为: %.3f%%" % acc)
                    print("Saving model......")
                    torch.save(net.state_dict(), '%s/resnet18_%03d.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH=%3d, Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    if acc > best_acc:
                        f3 = open("resnet_best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
                    writer.add_scalars('accuracy', {'train': train_acc, 'test': acc}, epoch)
            writer.close()
            print("Training Finished, TotalEPOCH=%d" % EPOCH)