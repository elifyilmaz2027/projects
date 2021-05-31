import torch.nn as nn

    
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()


        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()

        
        # Convolution 3
        self.cnn3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU()
        

        # Fully connected 1
        self.fc1 = nn.Linear(32 * 32 * 32, 10) 
        

    def forward(self, x):
        # Convolution 1
        output = self.cnn1(x)
        output = self.relu1(output)


        # Convolution 2 
        output = self.cnn2(output)
        output = self.relu2(output)

        
        # Convolution 3 
        output = self.cnn3(output)
        output = self.relu3(output)


        # Resize
        output = output.view(output.size(0), -1)

        # Linear function for fully connected layer
        output = self.fc1(output)

        return output

####### MODEL 2 ######

class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()


        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()

        
        # Convolution 3
        self.cnn3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU()
        
        

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 32 * 32, 1000) 
        self.fc2 = nn.Linear(1000, 100) 
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        # Convolution 1
        output = self.cnn1(x)
        output = self.relu1(output)


        # Convolution 2 
        output = self.cnn2(output)
        output = self.relu2(output)

        # Convolution 3 
        output = self.cnn3(output)
        output = self.relu3(output)

        # Resize
        output = output.view(output.size(0), -1)

        # Linear functions for fully connected layers
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output
    
#### MODEL 3 #####

class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(p=0.01)


        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(p=0.01)

        
        # Convolution 3
        self.cnn3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout2d(p=0.01)
        

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 32 * 32, 10) 
        

    def forward(self, x):
        # Convolution 1
        output = self.cnn1(x)
        output = self.relu1(output)
        output = self.dropout1(output)

        # Convolution 2 
        output = self.cnn2(output)
        output = self.relu2(output)
        output = self.dropout2(output)
    
        # Convolution 3 
        output = self.cnn3(output)
        output = self.relu3(output)
        output = self.dropout3(output)

        # Resize
        output = output.view(output.size(0), -1)

        # Linear function for fully connected layer
        output = self.fc1(output)
        

        return output


#### MODEL 4 #####

class Model4(nn.Module):
    def __init__(self):
        super(Model4, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.bachn1 = nn.BatchNorm2d(8)


        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.bachn2 = nn.BatchNorm2d(16)

        
        # Convolution 3
        self.cnn3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU()
        self.bachn3 = nn.BatchNorm2d(32)
        

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 32 * 32, 10) 
        

    def forward(self, x):
        # Convolution 1
        output = self.cnn1(x)
        output = self.relu1(output)
        output = self.bachn1(output)

        # Convolution 2 
        output = self.cnn2(output)
        output = self.relu2(output)
        output = self.bachn2(output)
    
        # Convolution 3 
        output = self.cnn3(output)
        output = self.relu3(output)
        output = self.bachn3(output)

        # Resize
        output = output.view(output.size(0), -1)

        # Linear function for fully connected layer
        output = self.fc1(output)
        

        return output
