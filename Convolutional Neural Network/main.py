import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pytorchtools1 import EarlyStopping
from model import Model1, Model2, Model3, Model4


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
test_batch = unpickle('cifar10_data/test_batch')

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
test_data = test_batch['data'] / 255
test_labels = np.array(test_batch['labels'])



#Reshape the data with the image data format
#We know that the CIFAR10 images are 32x32 RGB images.
data_1 = data_1.reshape(10000,32,32,3)
data_2 = data_2.reshape(10000,32,32,3)
data_3 = data_3.reshape(10000,32,32,3)
data_4 = data_4.reshape(10000,32,32,3)
data_5 = data_5.reshape(10000,32,32,3)
test_data = test_data.reshape(10000,32,32,3)


#Concatenate the train data by getting together batch data.
train_data = np.concatenate((data_1,data_2,data_3,data_4,data_5))
train_labels = np.concatenate((labels_1,labels_2,labels_3,labels_4,labels_5))

#For data augmentation part, convert numpy data to tensor.
#Also, we will use tensor data in madel training part.
tensor_train_data = torch.tensor(train_data)
tensor_test_data = torch.tensor(test_data)

##### Data Augmentation #######
# It includes random horizontal and vertical flips, normalizing data and random rotation with 20 degrees.

tensor_train_data.transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.RandomHorizontalFlip(p=0.1),
     transforms.RandomVerticalFlip(p=0.1),
     transforms.Normalize((0.30, 0.45, 0.50), (0.25, 0.36, 0.49)),
     transforms.RandomRotation(20)])

tensor_test_data.transform = transforms.Compose(
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

overall_test_data = tensor_test_data.float()
overall_test_data = overall_test_data.reshape(len(overall_test_data), 3, 32, 32)
test_labels = torch.tensor(test_labels)


print('##### MODEL 1 #####')
model1 = Model1()
optimizer1_1 = torch.optim.SGD(model1.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()


num_epochs = 20
train_loss_history1 = []
test_loss_history1 = []
batch_size = 100
iterations = int(len(overall_train_data) / batch_size)
torch.manual_seed(42)

# initialize the early_stopping
early_stopping = EarlyStopping(patience=1, verbose=True)

for epoch in range(num_epochs):
    train_total = 0
    train_correct = 0
    for i in range(0, iterations):
        i = i * batch_size
        
        train_images = overall_train_data[i:i + batch_size]
        
        train_images = train_images.requires_grad_()

        # Clear gradients w.r.t. parameters
        optimizer1_1.zero_grad()

        # Forward pass to get output/logits
        train_outputs = model1(train_images)
        
        train_labels = overall_train_labels[i:i + batch_size]
        
        # Get predictions from the maximum value
        _, train_predicted_labels = torch.max(train_outputs.data, 1)
        
        #To calculate accuracy, save the results
        train_total += train_labels.size(0)

        # Total correct predictions
        train_correct += (train_predicted_labels == train_labels).sum()


        # Calculate Loss: softmax --> cross entropy loss
        train_loss = criterion(train_outputs, train_labels)

        # Getting gradients w.r.t. parameters
        train_loss.backward()

        # Updating parameters
        optimizer1_1.step()
    
    #calculate accuracy 
    train_accuracy = train_correct / train_total
    # Load test images
    
    test_images = overall_test_data.requires_grad_()

    # Forward pass only to get logits/output
    test_outputs = model1(test_images)

    # Get predictions from the maximum value
    _, test_predicted_labels = torch.max(test_outputs.data, 1)

    # Calculate loss for test data
    test_loss = criterion(test_outputs, test_labels)
    
    # Total number of labels
    test_total = test_labels.size(0)

    # Total correct predictions for test data
    test_correct = (test_predicted_labels == test_labels).sum()
    test_accuracy = test_correct / test_total
    
    #Save losses during training for each epoch to see change of loss
    train_loss_history1.append(train_loss.item())
    test_loss_history1.append(test_loss.item())
    
    early_stopping(test_loss, model1)
    if early_stopping.early_stop:
            print("Early stopping")
            break
    
    # Print loss and accuracy for each epoch
    print('Epoch:{}/{}, Train Loss:{}, Train Accuracy:{}, Test Loss:{}, Test Accuracy:{}'.format(epoch, num_epochs, train_loss.item(), train_accuracy, test_loss.item(), test_accuracy))

#Plot Loss Changes 
plt.plot(train_loss_history1,'-o')
plt.plot(test_loss_history1,'-o')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'])
plt.title('Train and Test Loss Change for Model 2')
plt.show()


print('##### MODEL 2 #####')
model2 = Model2()
optimizer2_1 = torch.optim.SGD(model2.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()


num_epochs = 20
train_loss_history2 = []
test_loss_history2 = []
batch_size = 100
iterations = int(len(overall_train_data) / batch_size)
torch.manual_seed(42)

# initialize the early_stopping
early_stopping = EarlyStopping(patience=1, verbose=True)

for epoch in range(num_epochs):
    train_total = 0
    train_correct = 0
    for i in range(0, iterations):
        i = i * batch_size
        
        train_images = overall_train_data[i:i + batch_size]
        
        train_images = train_images.requires_grad_()

        # Clear gradients w.r.t. parameters
        optimizer2_1.zero_grad()

        # Forward pass to get output/logits
        train_outputs = model2(train_images)
        
        train_labels = overall_train_labels[i:i + batch_size]
        
        # Get predictions from the maximum value
        _, train_predicted_labels = torch.max(train_outputs.data, 1)
        
        #To calculate accuracy, save the results
        train_total += train_labels.size(0)

        # Total correct predictions
        train_correct += (train_predicted_labels == train_labels).sum()


        # Calculate Loss: softmax --> cross entropy loss
        train_loss = criterion(train_outputs, train_labels)

        # Getting gradients w.r.t. parameters
        train_loss.backward()

        # Updating parameters
        optimizer2_1.step()
    
    #calculate accuracy 
    train_accuracy = train_correct / train_total
    # Load test images
    
    test_images = overall_test_data.requires_grad_()

    # Forward pass only to get logits/output
    test_outputs = model2(test_images)

    # Get predictions from the maximum value
    _, test_predicted_labels = torch.max(test_outputs.data, 1)

    # Calculate loss for test data
    test_loss = criterion(test_outputs, test_labels)
    
    # Total number of labels
    test_total = test_labels.size(0)

    # Total correct predictions for test data
    test_correct = (test_predicted_labels == test_labels).sum()
    test_accuracy = test_correct / test_total
    
    #Save losses during training for each epoch to see change of loss
    train_loss_history2.append(train_loss.item())
    test_loss_history2.append(test_loss.item())
    
    early_stopping(test_loss, model2)
    if early_stopping.early_stop:
            print("Early stopping")
            break
    
    # Print loss and accuracy for each epoch
    print('Epoch:{}/{}, Train Loss:{}, Train Accuracy:{}, Test Loss:{}, Test Accuracy:{}'.format(epoch, num_epochs, train_loss.item(), train_accuracy, test_loss.item(), test_accuracy))

#Plot Loss Changes 
plt.plot(train_loss_history2,'-o')
plt.plot(test_loss_history2,'-o')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'])
plt.title('Train and Test Loss Change for Model 2')
plt.show()

print('##### MODEL 3 #####')
model3 = Model3()
optimizer3_1 = torch.optim.SGD(model3.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()


num_epochs = 20
train_loss_history3 = []
test_loss_history3 = []
batch_size = 100
iterations = int(len(overall_train_data) / batch_size)
torch.manual_seed(42)
# initialize the early_stopping
early_stopping = EarlyStopping(patience=1, verbose=True)

for epoch in range(num_epochs):
    train_total = 0
    train_correct = 0
    for i in range(0, iterations):
        i = i * batch_size
        
        train_images = overall_train_data[i:i + batch_size]
        
        train_images = train_images.requires_grad_()

        # Clear gradients w.r.t. parameters
        optimizer3_1.zero_grad()

        # Forward pass to get output/logits
        train_outputs = model3(train_images)
        
        train_labels = overall_train_labels[i:i + batch_size]
        
        # Get predictions from the maximum value
        _, train_predicted_labels = torch.max(train_outputs.data, 1)
        
        #To calculate accuracy, save the results
        train_total += train_labels.size(0)

        # Total correct predictions
        train_correct += (train_predicted_labels == train_labels).sum()


        # Calculate Loss: softmax --> cross entropy loss
        train_loss = criterion(train_outputs, train_labels)

        # Getting gradients w.r.t. parameters
        train_loss.backward()

        # Updating parameters
        optimizer3_1.step()
    
    #calculate accuracy 
    train_accuracy = train_correct / train_total
    # Load test images
    
    test_images = overall_test_data.requires_grad_()

    # Forward pass only to get logits/output
    test_outputs = model3(test_images)

    # Get predictions from the maximum value
    _, test_predicted_labels = torch.max(test_outputs.data, 1)

    # Calculate loss for test data
    test_loss = criterion(test_outputs, test_labels)
    
    # Total number of labels
    test_total = test_labels.size(0)

    # Total correct predictions for test data
    test_correct = (test_predicted_labels == test_labels).sum()
    test_accuracy = test_correct / test_total
    
    #Save losses during training for each epoch to see change of loss
    train_loss_history3.append(train_loss.item())
    test_loss_history3.append(test_loss.item())
    
    early_stopping(test_loss, model3)
    if early_stopping.early_stop:
            print("Early stopping")
            break
    
    # Print loss and accuracy for each epoch
    print('Epoch:{}/{}, Train Loss:{}, Train Accuracy:{}, Test Loss:{}, Test Accuracy:{}'.format(epoch, num_epochs, train_loss.item(), train_accuracy, test_loss.item(), test_accuracy))

#Plot Loss Changes 
plt.plot(train_loss_history3,'-o')
plt.plot(test_loss_history3,'-o')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'])
plt.title('Train and Test Loss Change for Model 3')
plt.show()



print('##### MODEL 4 #####')
model4 = Model4()
optimizer4_1 = torch.optim.SGD(model4.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()


num_epochs = 20
train_loss_history4 = []
test_loss_history4 = []
batch_size = 100
iterations = int(len(overall_train_data) / batch_size)
torch.manual_seed(42)
# initialize the early_stopping
early_stopping = EarlyStopping(patience=1, verbose=True)

for epoch in range(num_epochs):
    train_total = 0
    train_correct = 0
    for i in range(0, iterations):
        i = i * batch_size
        
        train_images = overall_train_data[i:i + batch_size]
        
        train_images = train_images.requires_grad_()

        # Clear gradients w.r.t. parameters
        optimizer4_1.zero_grad()

        # Forward pass to get output/logits
        train_outputs = model4(train_images)
        
        train_labels = overall_train_labels[i:i + batch_size]
        
        # Get predictions from the maximum value
        _, train_predicted_labels = torch.max(train_outputs.data, 1)
        
        #To calculate accuracy, save the results
        train_total += train_labels.size(0)

        # Total correct predictions
        train_correct += (train_predicted_labels == train_labels).sum()


        # Calculate Loss: softmax --> cross entropy loss
        train_loss = criterion(train_outputs, train_labels)

        # Getting gradients w.r.t. parameters
        train_loss.backward()

        # Updating parameters
        optimizer4_1.step()
    
    #calculate accuracy 
    train_accuracy = train_correct / train_total
    # Load test images
    
    test_images = overall_test_data.requires_grad_()

    # Forward pass only to get logits/output
    test_outputs = model4(test_images)

    # Get predictions from the maximum value
    _, test_predicted_labels = torch.max(test_outputs.data, 1)

    # Calculate loss for test data
    test_loss = criterion(test_outputs, test_labels)
    
    # Total number of labels
    test_total = test_labels.size(0)

    # Total correct predictions for test data
    test_correct = (test_predicted_labels == test_labels).sum()
    test_accuracy = test_correct / test_total
    
    #Save losses during training for each epoch to see change of loss
    train_loss_history4.append(train_loss.item())
    test_loss_history4.append(test_loss.item())
    
    early_stopping(test_loss, model4)
    if early_stopping.early_stop:
            print("Early stopping")
            break
    
    # Print loss and accuracy for each epoch
    print('Epoch:{}/{}, Train Loss:{}, Train Accuracy:{}, Test Loss:{}, Test Accuracy:{}'.format(epoch, num_epochs, train_loss.item(), train_accuracy, test_loss.item(), test_accuracy))

#Plot Loss Changes 
plt.plot(train_loss_history4,'-o')
plt.plot(test_loss_history4,'-o')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'])
plt.title('Train and Test Loss Change for Model 4 with SGD Optimizer')
plt.show()


