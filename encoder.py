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
        out = out.permute(0, 2, 1)
        out = self.norm(out)
        
        out = out.permute(0, 2, 1)
        
        out = self.elu(out)
        
        
        out = self.conv2(out)
        out=out+residual
        out = self.elu(out)
        
    
        return out



x=torch.randn(4,32,100)
model=ResUnit(32)

#o=model(x)
#print(o)
#print(o.shape)

class DownSamplerLayer(nn.Module):
    def __init__(self,in_channels:int,stride:int):
        super().__init__()
        kernel_size=2*stride
        out_channels=2*in_channels
        padding = (kernel_size - stride) // 2
        self.conv=nn.Conv1d(in_channels=in_channels,out_channels=out_channels,stride=stride,kernel_size=kernel_size,padding=padding)
        self.norm=nn.LayerNorm(out_channels)
        self.elu=nn.ELU()
    def forward(self,x):
        x=self.conv(x)
        x=x.permute(0,2,1)
        x=self.norm(x)
        x=x.permute(0,2,1)
        x=self.elu(x)
        return x 



#d=DownSamplerLayer(32,4)
#print(d(o).shape)

## 2xin_channel out from conv block  

class ConvBlock(nn.Module):
    def __init__(self,in_channels:int,stride:int):
        super().__init__()
        self.resunit=ResUnit(in_channels)
        self.downsampler=DownSamplerLayer(in_channels=in_channels,stride=stride)
    def forward(self,x):
        x=self.resunit(x)
        x=self.downsampler(x)
        return x 




class Encoder(nn.Module):
    def __init__(self,in_channels:int,hidden_channels:int,out_channels:int,n_conv_blocK:int):
        super().__init__()
        self.strides=[2,4,5,8]
        self.conv1=nn.Conv1d(in_channels=in_channels,out_channels=hidden_channels,kernel_size=7)
        blocks=[]
        out=hidden_channels
        for i in range(n_conv_blocK):
            s=self.strides[i]
            blocks.append(ConvBlock(in_channels=out,stride=s))
            out=2*out

        self.conv_blocks=nn.Sequential(*blocks)
            
        self.lstm=nn.LSTM(input_size=out,hidden_size=hidden_channels,num_layers=2,batch_first=True)
        self.conv2=nn.Conv1d(in_channels=hidden_channels,out_channels=out_channels,kernel_size=7) 
        
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv_blocks(x)
        x=x.permute(0,2,1)
        x,_=self.lstm(x)
        x=x.permute(0,2,1)
        x=self.conv2(x)
        x=x.permute(0,2,1)
        return x




 
      


E=Encoder(in_channels=1,hidden_channels=32,out_channels=128,n_conv_blocK=4)
x=torch.randn(2,1,24000)
o=E(x)
print(o)
print(o.shape)
