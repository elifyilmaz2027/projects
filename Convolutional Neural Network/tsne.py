import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model import Model4


def unpickle(file):
    with open(file, 'rb') as fo:
        a = pickle.load(fo, encoding='latin1')
    return a

#import all data to train model
batches_meta_data = unpickle('cifar10_data/batches.meta')
label_names = batches_meta_data['label_names']

data_batch_1 = unpickle('cifar10_data/data_batch_1')
data_batch_2 = unpickle('cifar10_data/data_batch_2')
data_batch_3 = unpickle('cifar10_data/data_batch_3')
data_batch_4 = unpickle('cifar10_data/data_batch_4')
data_batch_5 = unpickle('cifar10_data/data_batch_5')


#take image data and labels from different batch files.
#for batch 1
data_1 = data_batch_1['data'] / 255
labels_1 = np.array(data_batch_1['labels'])
#for batch 2
data_2 = data_batch_2['data'] / 255
labels_2 = np.array(data_batch_2['labels'])
#for batch 3
data_3 = data_batch_3['data'] / 255
labels_3 = np.array(data_batch_3['labels'])
#for batch 4
data_4 = data_batch_4['data'] / 255
labels_4 = np.array(data_batch_4['labels'])
#for batch 5
data_5 = data_batch_5['data'] / 255
labels_5 = np.array(data_batch_5['labels'])




#Reshape the data with the image data format
#We know that the CIFAR10 images are 32x32 RGB images.
data_1 = data_1.reshape(10000,32,32,3)
data_2 = data_2.reshape(10000,32,32,3)
data_3 = data_3.reshape(10000,32,32,3)
data_4 = data_4.reshape(10000,32,32,3)
data_5 = data_5.reshape(10000,32,32,3)


#Concatenate the train data by getting together batch data.
train_data = np.concatenate((data_1,data_2,data_3,data_4,data_5))
train_labels = np.concatenate((labels_1,labels_2,labels_3,labels_4,labels_5))

#For data augmentation part, convert numpy data to tensor.
#Also, we will use tensor data in madel training part.
tensor_train_data = torch.tensor(train_data)

##### Data Augmentation #######
# It includes random horizontal and vertical flips, normalizing data and random rotation with 20 degrees.

tensor_train_data.transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.RandomHorizontalFlip(p=0.1),
     transforms.RandomVerticalFlip(p=0.1),
     transforms.Normalize((0.30, 0.45, 0.50), (0.25, 0.36, 0.49)),
     transforms.RandomRotation(20)])



#Prepare data to train model.
#Here, reshape data since the format of input image is (channel, height, width) in pytorch.
overall_train_data = tensor_train_data.float()
overall_train_data = overall_train_data.reshape(len(overall_train_data), 3, 32, 32)
overall_train_labels = torch.tensor(train_labels)



model4 = Model4()
optimizer4_1 = torch.optim.SGD(model4.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()


num_epochs = 4
batch_size = 100
iterations = int(len(overall_train_data) / batch_size)
torch.manual_seed(42)

for epoch in range(num_epochs):
    train_images = overall_train_data
    train_images = train_images.requires_grad_()
    # Clear gradients w.r.t. parameters
    optimizer4_1.zero_grad()
    # Forward pass to get output/logits
    train_outputs = model4(train_images)
    
    # Get predictions from the maximum value
    _, predicted_labels = torch.max(train_outputs.data, 1)
        
    f = train_outputs.detach().numpy()
    tsne = TSNE(n_components=2, verbose=1)
    t1 = tsne.fit_transform(f)
    fig, ax = plt.subplots()
    groups = predicted_labels.numpy()
    for g in np.unique(groups):
        i = np.where(groups == g)
        ax.scatter(t1[i,0], t1[i,1], label=g)
    ax.legend(['airplane', 'automobile','bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
    plt.show()
    
    train_labels = overall_train_labels

    # Calculate Loss: softmax --> cross entropy loss
    train_loss = criterion(train_outputs, train_labels)

    # Getting gradients w.r.t. parameters
    train_loss.backward()

    # Updating parameters
    optimizer4_1.step()



