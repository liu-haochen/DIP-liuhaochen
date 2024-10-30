import torch.nn as nn

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
         # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        
        ### FILL: add more CONV Layers
        
        # Decoder (Deconvolutional Layers)
        ### FILL: add ConvTranspose Layers
        ### None: since last layer outputs RGB channels, may need specific activation function

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),  # Input channels: 3 (RGB), Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),  # Output channels: 16
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # Output channels: 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output channels: 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output channels: 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output channels: 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

        )
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Output channels: 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),


            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output channels: 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output channels: 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # Output channels: 16
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  # Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1)   # Output channels: 3 (RGB)
        )       
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoder forward pass
        encoded = self.encoder(x)

        # Decoder forward pass
        output = self.decoder(encoded)
        output = self.tanh(output)
        ### FILL: encoder-decoder forward pass

        #output = ...
        
        return output
    