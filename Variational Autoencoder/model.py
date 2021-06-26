import torch
from torch import nn
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, num_layers):
        super(VAE, self).__init__()
        
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size1 = hidden_size1 
        self.hidden_size2 = hidden_size2 
        self.hidden_size3 = hidden_size3 
        self.hidden_size4 = hidden_size4

        #encoder part
        #at first, we use one layer lstm.
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size1,
                            num_layers=num_layers, batch_first=True) 
        #Then, we can apply relu activation fuction and fully connected layers.
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(hidden_size1, hidden_size4)
        self.fc2 = nn.Linear(hidden_size1, hidden_size4)
        # decoder part
        self.conv1 = nn.ConvTranspose2d(hidden_size4, hidden_size3, 5, stride=(3,1))
        self.relu2 = nn.ReLU()
        self.bachn1 = nn.BatchNorm2d(hidden_size3)
        self.pool1 = nn.MaxPool2d((3,2))
        self.dropout1 = nn.Dropout2d(p=0.01)
        
        self.conv2 = nn.ConvTranspose2d(hidden_size3, hidden_size2, 5, stride=(3,1))
        self.relu3 = nn.ReLU()
        self.bachn2 = nn.BatchNorm2d(hidden_size2)
        self.pool2 = nn.MaxPool2d((3,2))
        self.dropout2 = nn.Dropout2d(p=0.01)
        
        self.conv3 = nn.ConvTranspose2d(hidden_size2, 1, 5, stride=(3,1))
        self.relu4 = nn.ReLU()
        self.bachn3 = nn.BatchNorm2d(1)
        self.pool3 = nn.MaxPool2d((3,2))
        self.dropout3 = nn.Dropout2d(p=0.01)

        self.fc3 =  nn.Linear(3, 28) #fully connected layer for decoder part
        

    
    def encoder(self, data):
        h_0 = torch.zeros(self.num_layers, data.size(0), self.hidden_size1).to(device) #hidden state
        c_0 = torch.zeros(self.num_layers, data.size(0), self.hidden_size1).to(device) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(data, (h_0, c_0)) 
        
        output = self.relu1(output) 
        
        mu = self.fc1(output)
        var = self.fc2(output)
        return mu, var
    
    def sampling(self, mu, var):
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        # Convolution 1
        output = self.conv1(z)
        output = self.relu2(output)
        output = self.bachn1(output)
        output = self.pool1(output)
        output = self.dropout1(output)

        # Convolution 2 
        output = self.conv2(output)
        output = self.relu3(output)
        output = self.bachn2(output)
        output = self.pool2(output)
        output = self.dropout2(output)

        # Convolution 3 
        output = self.conv3(output)
        output = self.relu4(output)
        output = self.bachn3(output)
        output = self.pool3(output)
        output = self.dropout3(output)
        
        output = F.sigmoid(self.fc3(output))
        
        return  output
    
    def forward(self, data):
        mu, var = self.encoder(data)
        z = self.sampling(mu, var)
        z = z.reshape(len(z),256,28,1)
        return self.decoder(z), mu, var



# return reconstruction loss with regularization term with KL divergence
def loss_function(recon_data, data, mu, var):
    data = data.detach()
    BCE = F.binary_cross_entropy(recon_data.view(-1,28*28), data.view(-1, 28*28), reduction='mean')
    KLD = -0.5 * torch.mean(1 + var - mu.pow(2) - var.exp())
    return BCE + KLD, KLD



