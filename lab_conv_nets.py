import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torchvision

import matplotlib.pyplot as plt
import numpy as np


# functions to show an image
def imshow(img):
    print(f'{img.shape=}')
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    print(f'{npimg.shape=}')
    plt.imshow(npimg)
    plt.show()


def todo_1():
    convert_to_tensor = torchvision.transforms.ToTensor()

    train_cifar10 = torchvision.datasets.CIFAR10('CIFAR10', download=True, train=True, transform=convert_to_tensor)
    test_cifar10 = torchvision.datasets.CIFAR10('CIFAR10', train=False, transform=convert_to_tensor)

    print(f'{len(train_cifar10)=}')
    print(f'{train_cifar10[0]=}')
    print(f'{train_cifar10[0][0][:, 0, 0]=}')
    print(f'{train_cifar10[0][0].shape=}')
    print(f'{train_cifar10[0][1]=}')

    cifar10_classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_cifar10), size=(1,)).item()
        img, label = train_cifar10[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(cifar10_classes[label])
        plt.axis("off")
        plt.imshow(img.numpy().transpose((1, 2, 0)), cmap="gray")
    plt.show()

    batch_size = 8
    trainloader = torch.utils.data.DataLoader(train_cifar10, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_cifar10, batch_size=batch_size, shuffle=False, num_workers=2)

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % cifar10_classes[labels[j]] for j in range(batch_size)))


def todo_2():
    train_cifar10 = torchvision.datasets.CIFAR10('CIFAR10', download=True, train=True, transform=torchvision.transforms.ToTensor())
    test_cifar10 = torchvision.datasets.CIFAR10('CIFAR10', train=False, transform=torchvision.transforms.ToTensor())

    # Before
    print(f'{len(train_cifar10)=}')
    print(f'{len(test_cifar10)=}')

    # Random split
    train_set_size = int(len(train_cifar10) * 0.8)
    valid_set_size = len(train_cifar10) - train_set_size
    train_cifar10, valid_cifar10 = torch.utils.data.random_split(train_cifar10, [train_set_size, valid_set_size])

    # After
    print('=' * 30)
    print(f'{len(train_cifar10)=}')
    print(f'{len(test_cifar10)=}')
    print(f'{len(valid_cifar10)=}')

    train_loader = torch.utils.data.DataLoader(train_cifar10, batch_size=512, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(valid_cifar10, batch_size=512, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_cifar10, batch_size=512, num_workers=2)

    cifar10_classes = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    image_size = 3 * 32 * 32

    model = nn.Sequential(
        # Warstwa "spłaszczająca", która odpowiada za rozwinięcie wszystkich wymiarów tensora wejściowego do jednowymiarowego,
        # ciągłego wektora. Wymagana ze względu na kolejną warstwę.
        nn.Flatten(),

        # Warstwa w pełni połączona - musi przyjąć in_features (tutaj: rozmiar obrazu) wartości i zwrócić out_features wartości
        nn.Linear(in_features=image_size, out_features=512),
        # Warstwa, która warstwą nie jest - aplikuje jedynie funkcję aktywacji ReLU. Argument inplace=True jest optymalizacją - modyfikuje tensor,
        # który otrzymuje zamiast tworzyć nowy.
        nn.ReLU(inplace=True),

        nn.Linear(in_features=512, out_features=256),
        nn.ReLU(inplace=True),

        nn.Linear(in_features=256, out_features=128),
        nn.ReLU(inplace=True),

        # Wartwa "końcowa", mająca tyle jednostek ile przewidywanych klas
        nn.Linear(in_features=128, out_features=len(cifar10_classes)),
        # NIE używamy funkcji aktywacji softmax, która normalizuje
        # wyjścia sieci tak, że sumują się one do 1, a wartości poszczególnych jednostek możemy traktować jako prawdopodobieństwa klas.
        # Podczas uczenia softmax zastosuje za nas funkcja kosztu "CrossEntropyLoss", jednak należy pamiętać
        # że ostatecznie wyjścia sieci nie będą znormalizowane.
    )

    learning_rate = 1e-1  # 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    device = torch.device('cuda')  # skorzystajmy z dobrodziejstw treningu z wykorzystaniem GPU
    model = model.to(device)  # przenieśmy nasz model na GPU

    for epoch in range(10):  # przejdźmy po naszym zbiorze uczącym kilka razy
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # wczytajmy wsad (batch) wejściowy: dane i etykiety
            inputs, labels = data

            # przenieśmy nasze dane na GPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            # wyzerujmy gradienty parametrów
            optimizer.zero_grad()

            # propagacja w przód, w tył i optymalizacja
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # drukowanie statystyk
            running_loss += loss.item()
            if i % 10 == 9:  # drukujmy co dziesiąty batch
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    print('Finished Training')




def main():
    # todo_1()
    todo_2()


if __name__ == '__main__':
    main()
