import torch
from torch import nn 
import torch.nn.functional as F 



class ResUnit(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=channels, 
            out_channels=channels, 
            kernel_size=3, 
            padding=1, 
            bias=False
        )
        
        
        self.norm = nn.LayerNorm(normalized_shape=channels)
        self.elu = nn.ELU()
        
    
        self.conv2 = nn.Conv1d(
            in_channels=channels, 
            out_channels=channels, 
            kernel_size=3, 
            padding=1, 
            bias=False
        )

    def forward(self, x):
        
        residual = x 
        out = self.conv1(x)
        out=out.permute(0,2,1)
        out = self.norm(out)
        
        out = self.elu(out)
        out=out.permute(0,2,1)
        
        out = self.conv2(out)
        out=out+residual
        out = self.elu(out)
        
    
        return out

class UpSamplerLayer(nn.Module):
    def __init__(self,in_channels:int,stride:int):
        super().__init__()
        kernel_size=2*stride
        out_channels=in_channels//2
        self.conv=nn.ConvTranspose1d(in_channels=in_channels,out_channels=out_channels,stride=stride,kernel_size=kernel_size,padding=stride//2,output_padding=stride%2)
        self.norm=nn.LayerNorm(out_channels)
        self.elu=nn.ELU()
    def forward(self,x):
        x=self.conv(x) 
        x=x.permute(0,2,1)
        x=self.norm(x)
        x=self.elu(x)
        x=x.permute(0,2,1)
        return x 





class ReverseConvBlock(nn.Module):
    def __init__(self,in_channels:int,stride:int):
        super().__init__()
        self.resunit=ResUnit(in_channels//2)
        self.upsampler=UpSamplerLayer(in_channels=in_channels,stride=stride)
    def forward(self,x):     
        x=self.upsampler(x)
        x=self.resunit(x)
        return x 




class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_channels: int, out_channels: int, n_conv_blocks: int):
        super().__init__()
        self.strides = [8, 5, 4, 2]
        deepest_channels = hidden_channels * (2 ** n_conv_blocks)

        # 1. Reverse of encoder's final conv2
        self.conv1 = nn.Conv1d(
            in_channels=latent_dim, 
            out_channels=hidden_channels, 
            kernel_size=7, 
            padding=3
        )

        # 2. Reverse of encoder's LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_channels, 
            hidden_size=deepest_channels, 
            num_layers=2, 
            batch_first=True
        )
        self.lstm_proj = nn.Conv1d(
            in_channels=deepest_channels, 
            out_channels=deepest_channels, 
            kernel_size=1
        )

        # 3. Upsampling blocks
        blocks = []
        curr_channels = deepest_channels
        for i in range(n_conv_blocks):
            s = self.strides[i]
            blocks.append(ReverseConvBlock(in_channels=curr_channels, stride=s))
            curr_channels = curr_channels // 2

        self.conv_blocks = nn.Sequential(*blocks)

        # 4. Reverse of encoder's initial conv1 (back to 1-channel audio)
        self.conv2 = nn.Conv1d(
            in_channels=hidden_channels, 
            out_channels=out_channels, 
            kernel_size=7, 
            padding=3
        )

    def forward(self, x):
    
        x = x.permute(0, 2, 1)
        
        x = self.conv1(x)
        
        # LSTM expects [Batch, Time, Channels]
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)  # -> [Batch, Channels, Time]
        
        x = self.lstm_proj(x)
        x = self.conv_blocks(x)
        x = self.conv2(x)
        
        return x






decoder=Decoder(128,64,1,4)
c=decoder(torch.randn(2,4,128))
print(c)
print(c.shape)
