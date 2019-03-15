from model import *
from data import *
from transform import *
def loadnet(model_path, device):
    weights = torch.load(model_path)['net']
    net = MobileNetV2()
    net.load_state_dict(weights)
    net.to(device)
    return net

def test_img(img_path, model_path ,device):
    weights = torch.load(model_path)['net']
    net = loadnet(model_path, device)
    img = io.imread(img_path)
    test_img = transform.resize(img, (64,64))
    test_img = np.float32(test_img.transpose((2,0,1)))
    test_img = torch.from_numpy(test_img)
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    image = normalize(test_img)
    image = image.view(-1, 3, 64, 64).to(device)
    outputs = net(image)
    outputs = (outputs.cpu().detach().numpy() * 64).reshape([68,2])
    outputs = outputs.astype(int)
    plt.figure()
    show = transform.resize(img, (64,64))
    plt.imshow(show)
    plt.scatter(outputs[:,0],outputs[:,1],s=10, marker='.', c='r')
    
def test(net, testset, device, writer, epoch):
    testloader = DataLoader(testset, 100, shuffle=False, num_workers=4)
    mean_error = 0
    with torch.no_grad():
           for i_batch, sample_batched in enumerate(testloader):
                inputs, labels = sample_batched['image'].to(device), sample_batched['landmarks'].to(device)
                outputs = net(inputs)
                labels = labels.view(-1, 136)
                mean_error += torch.sum((outputs - labels)**2)
                if i_batch == 0:
                    outputs = (outputs.view(-1,68,2) * 64).cpu().detach()
                    batch = {'image':sample_batched['image'], 'landmarks':outputs}
                    show_landmarks_batch(batch)

    mean_error = mean_error / len(testset)
    writer.add_scalar('Mean_error' ,mean_error ,epoch)
    return mean_error

def train(device, net, criterion, optimizer, trainset, testset, epoch, batch_size, save_path, writer):
    net.to(device)
    running_loss, min_error = 0.0, 10000
    trainloader = DataLoader(trainset, batch_size, shuffle=True, num_workers=4)
    
    dataset_length = len(trainset)
    for i in range(epoch):
        running_loss = 0.0
        for i_batch, sample_batched in enumerate(trainloader):
            inputs, labels = sample_batched['image'].to(device), sample_batched['landmarks'].to(device)
            outputs = net(inputs)
            labels = labels.view(-1, 136)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i_batch % 500 == 499:
                print('epoch:'+ str(i) + ' iteration:' + str(i_batch) + ' loss:' + str(running_loss/500.0))
                writer.add_scalar('Loss',running_loss/500.0,i_batch + i * dataset_length)
                running_loss = 0
        mean_error = test(net, testset, device, writer, i)
        if mean_error < min_error:
            state = {'net':net.state_dict()}
            torch.save(state, save_path)
    writer.close()

    