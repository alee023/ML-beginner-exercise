import torch, torchvision #
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = torchvision.datasets.MNIST(root="./", train=True, download=True)
test_dataset = torchvision.datasets.MNIST(root="./", train=False, download=True)

# filtering to keep only 3s and 7s
indices0 = ( train_dataset.targets == torch.tensor( 3 )) | ( train_dataset.targets == torch.tensor( 7 ))
train_dataset.data, train_dataset.targets = train_dataset.data[ indices0 ], train_dataset.targets[ indices0 ]
indices1 = ( test_dataset.targets == torch.tensor( 3 )) | ( test_dataset.targets == torch.tensor( 7 ))
test_dataset.data, test_dataset.targets = test_dataset.data[ indices1 ], test_dataset.targets[ indices1 ]

# normalization and resizing for resnet50 -- 224x224
data_transforms = transforms.Compose([ transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize((0.1397,),(0.3081,))]) 

train_dataset.transform = data_transforms
test_dataset.transform = data_transforms

# batch size = 128
train_loader = torch.utils.data.DataLoader( train_dataset, batch_size = 128, shuffle = True )
test_loader = torch.utils.data.DataLoader( test_dataset, batch_size = 512, shuffle = True )

# a, b = next( iter(train_loader ))
# print( a.shape, b.shape )

class Net( nn.Module ) :
    def __init__( self ) :
        super( Net, self ).__init__()
        
        self.model = models.resnet50( pretrained = True )
        # changed in_channels from 3 to 1 bc images are black and white 
        self.model.conv1 = nn.Conv2d( 1, 64, kernel_size = 7, stride = 2, padding = 3, bias = False )
        
        # binary classifier -> 2 out_features
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear( num_ftrs, 2 )
        
    def forward( self, x ):
        return self.model( x )
    
model = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD( model.parameters(), lr=0.0001, momentum=0.9 )

for epoch in range( 20 ) :
    running_loss = 0.0

    num_correct = 0
    for x, y in enumerate( train_loader ):
        optimizer.zero_grad()
        y_pred = model( x )
        
        loss = criterion( y_pred, y )
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        y_pred = y_pred.data.max(1,keepdim=True)[1]
        num_correct += y_pred.eq( y.data.view_as(y_pred)).sum()
    running_loss /= len(train_loader) #Divide by the number of batches to get the average sample error
    print("Epoch {}: Training Loss: {:.5f} Accuracy: {}/{}".format(epoch+1, running_loss, num_correct, len(train_dataset)))
    
    