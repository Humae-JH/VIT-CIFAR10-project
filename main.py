from model import *

SEED = 1234


num_workers = 0
IMAGE_SIZE = 32
PATCH_SIZE = 8
Emb_dimension = 64
#NUM_HEAD=16
#Emb_dimension = 320
NUM_HEAD = 8
learning_rate = 0.0002
epoch = 100

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(0.5),
    normalize,
    ])

batch_size = 32

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



def showDataloader(dataloader):

    xb, yb = next(iter(dataloader))

    row_num, col_num = int(batch_size / 8), 8
    fig, axs = plt.subplots(row_num, col_num, figsize=(10, 10 * row_num / col_num))

    for row_idx in range(row_num):
        for col_idx in range(col_num):
            ax = axs[row_idx][col_idx]
            i = col_idx * row_num + row_idx

            class_index = yb[i].item()
            class_label = classes[class_index].split(",")[0]
            img = xb[i].permute(1, 2, 0)
            ax.title.set_text(class_label)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.imshow(img)
    plt.tight_layout(pad=0.5)
    plt.show()

def TrainVit():
    model = VisionTransformer(IMAGE_SIZE, PATCH_SIZE, Emb_dimension, NUM_HEAD, device, learning_rate).to(device)

    summary(model, (3, IMAGE_SIZE, IMAGE_SIZE), batch_size)
    model.Train(trainloader, epoch)
    accuracy, f1 = model.Vaild(testloader)

    torch.save(model.state_dict(), f"vit_model(img_size{IMAGE_SIZE},pat_size{PATCH_SIZE},Emb{Emb_dimension},Head{NUM_HEAD},ep{epoch},ACC{accuracy}.pth")
    return

def ValidVit(pth):
    model = VisionTransformer(IMAGE_SIZE, PATCH_SIZE, Emb_dimension, NUM_HEAD, device, learning_rate).to(device)
    model.load_state_dict(torch.load(pth))
    summary(model, (3, IMAGE_SIZE, IMAGE_SIZE), batch_size)

    accuracy, f1 = model.Vaild(testloader)
def TrainCNN_L():


    model = SimpleCNN_L(IMAGE_SIZE, 10, device, learning_rate).to(device)


    summary(model, (3, IMAGE_SIZE, IMAGE_SIZE), batch_size)
    model.Train(trainloader, epoch)
    accuracy, f1 = model.Vaild(testloader)

    torch.save(model.state_dict(), f"cnn_model(img_size{IMAGE_SIZE},pat_size{PATCH_SIZE},Emb{Emb_dimension},Head{NUM_HEAD},ep{epoch},ACC{accuracy}.pth")

def TrainCNN_S():


    model = SimpleCNN_S(IMAGE_SIZE, 10, device, learning_rate).to(device)


    summary(model, (3, IMAGE_SIZE, IMAGE_SIZE), batch_size)
    model.Train(trainloader, epoch)
    accuracy, f1 = model.Vaild(testloader)

    torch.save(model.state_dict(), f"cnn_model(img_size{IMAGE_SIZE},pat_size{PATCH_SIZE},Emb{Emb_dimension},Head{NUM_HEAD},ep{epoch},ACC{accuracy}.pth")

def ValidCNN_L(pth):
    model = SimpleCNN_L(IMAGE_SIZE, 10, device, learning_rate).to(device)
    model.load_state_dict(torch.load(pth))
    summary(model, (3, IMAGE_SIZE, IMAGE_SIZE), batch_size)

    accuracy, f1 = model.Vaild(testloader)

def ValidCNN_S(pth):
    model = SimpleCNN_S(IMAGE_SIZE, 10, device, learning_rate).to(device)
    model.load_state_dict(torch.load(pth))
    summary(model, (3, IMAGE_SIZE, IMAGE_SIZE), batch_size)

    accuracy, f1 = model.Vaild(testloader)

#TrainVit()
ValidVit("Vit_S_model.pth")
#ValidVit("Vit_L_model.pth")
#TrainCNN_S()
#ValidCNN_S("CNN_S_model.pth")
#ValidCNN_L("CNN_L_model.pth")