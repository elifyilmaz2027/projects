from torchvision import datasets
from torchvision.transforms import ToTensor
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from model import VAE
from model import loss_function

train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor(),
    download = True)

print(train_data)
print(test_data)
print(train_data.data.size())
print(train_data.targets.size())
print(test_data.data.size())
print(test_data.targets.size())

#plot some of the train data
figure = plt.figure(figsize=(10, 8))
plt.title('Random Images from MNIST Data')
plt.axis("off")
cols, rows = 10, 10
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data.data), size=(1,)).item()
    img = train_data.data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

sequence_length = 28
input_size = 28
hidden_size1 = 32
hidden_size2 = 64
hidden_size3 = 128
hidden_size4 = 256
num_layers = 1
batch_size = 100
num_epochs = 50
learning_rate = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#normalize data
overall_train_data = train_data.data / 255
overall_train_labels = train_data.targets


# build model
model_vae = VAE(input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, num_layers).to(device)
print('Variational Autoencoder Model with LSTM as encoder and CNN as decoder')

optimizer = torch.optim.Adam(model_vae.parameters(), lr = learning_rate)
train_loss_history = []
train_regularization_terms = []
batch_size = 100
iterations = int(len(overall_train_data) / batch_size)
torch.manual_seed(42)

for epoch in range(num_epochs):
    train_reconstruction_images = []
    for i in range(0, iterations):
        i = i * batch_size
        
        train_images = overall_train_data[i:i + batch_size].to(device)
        
        train_images = train_images.requires_grad_()

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        recon_train_data, mu, var = model_vae(train_images)
        recon_train_data1 = recon_train_data.reshape(len(recon_train_data),28,28)
        recon_train_data2 = recon_train_data1.cpu().detach().numpy()
        train_reconstruction_images.append(recon_train_data2)
        
        # Calculate Loss and save regularization term too see change
        train_loss, train_regularization_term = loss_function(recon_train_data1, train_images, mu, var)
        
        
        # Getting gradients w.r.t. parameters
        train_loss.backward()

        # Updating parameters
        optimizer.step()
    
    
    #Save losses during training for each epoch to see change of loss
    train_loss_history.append(train_loss.item())
    train_regularization_terms.append(train_regularization_term.item())
    
    
    # Print loss and accuracy for each epoch
    print('Epoch:{}/{}, Train_Loss:{}, Train_Reg_Term:{}'.format(epoch, num_epochs, train_loss.item(), train_regularization_term.item()))




#Plot Loss Change
plt.plot(train_loss_history,'-o')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend(['Train'])
plt.title('Train Loss Change for VAE Model')
plt.show()

#Plot Regularization Term
plt.plot(train_regularization_terms,'-o')
plt.xlabel('epoch')
plt.ylabel('Regularization Term with KL Divergence')
plt.legend(['Train'])
plt.title('Change of Train Regularization Term with KL Divergence for VAE Model')
plt.show()

train_reconstruction_images = np.concatenate(train_reconstruction_images, axis=0)


model = {'train_reconstruction_images': train_reconstruction_images, 
         'train_loss_history': train_loss_history,
         'train_regularization_terms': train_regularization_terms}

# save the model to disk
filename = 'model.pk'
pickle.dump(model, open(filename, 'wb'))

