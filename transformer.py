import torch
from torch import nn
import torch.nn.functional as F 
from encoder import Encoder
#from rvq import ResidualVectorQuantization
import math 

def sinusoidual_positional(T,d,offset=0):
    # postion:[0,1,2,3...N]
    pos=torch.arange(0,T).float().unsqueeze(1) + offset
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d)) 
    pe=torch.zeros(T,d)
    pe[:,0::2]=torch.sin(pos * div)
    pe[:,1::2]=torch.cos(pos * div)

    return pe
    

class CodeBookTransformer(nn.Module):
    def __init__(self,n_codebook:int,codebook_size:int,embed_dim:int,n_head:int,n_layer:int,max_offset:int,feedf_dim:int,window_size:int):
        super().__init__()
        self.codebook_size=codebook_size
        self.n_codebook=n_codebook
        self.max_offset=max_offset
        self.embed_dim=embed_dim
        self.window_size=window_size

        

        # one mebed per codebook 
        self.embeds = nn.ModuleList([nn.Embedding(codebook_size + 1, embed_dim) for _ in range(n_codebook)])
        encoder_layer=nn.TransformerEncoderLayer(d_model=embed_dim,nhead=n_head,dim_feedforward=feedf_dim,dropout=0.0,batch_first=True,norm_first=True,activation='gelu')

        self.transformer=nn.TransformerEncoder(encoder_layer=encoder_layer,num_layers=n_layer)


        # one layer for one coebook 
        self.heads=nn.ModuleList([nn.Linear(embed_dim,codebook_size) for _ in range(n_codebook)])

    def forward(self,x:torch.Tensor):
        batch_size,n_codebook,seq=x.shape
        # adding the 1024 to every tokem as eos
        start_tokens=torch.full((batch_size,n_codebook,1),self.codebook_size,dtype=x.dtype)
        # here is [1024,...] is the input && [orginal one is target model to predict the last tokem]

        prev=torch.cat([start_tokens,x[:,:,:-1]],dim=-1) # adding to last tensor 
        out = torch.sum(torch.stack([embd(prev[:, i]) for i, embd in enumerate(self.embeds)]), dim=0)  
        offset=torch.randint(0,self.max_offset,(1,)).item() if self.training else 0
        out = out + sinusoidual_positional(seq, out.shape[-1], offset)
        mask=torch.triu(torch.ones(seq,seq),diagonal=1).bool()
        h = self.transformer(out, mask=mask)                           # causal, ~3.5 s window
        return torch.stack([head(h) for head in self.heads], dim=1) 



    def loss(self,indices:torch.Tensor):
        logits=self(indices)
        return F.cross_entropy(logits.reshape(-1,self.codebook_size),indices.reshape(-1))



     
def estimated_kbps(ce_nats, n_q, frames_per_sec=75):

    """What the arithmetic coder would need, est imated from the cross-entropy."""
    return ce_nats / math.log(2) * n_q * frames_per_sec / 1000



codes = torch.randint(0, 1024, (2, 4, 6))
model = CodeBookTransformer(
    n_codebook=4, 
    codebook_size=1024,  # <--- Change this from 4 to 1024
    embed_dim=128, 
    n_head=2, 
    n_layer=2, 
    max_offset=40, 
    feedf_dim=256, 
    window_size=3
)
#Once you update codebook_size=1024, your embedding tables will correctly accommodate all token indices from your RVQ up to 1024, plus the special start token!

y=model(codes)
print(y)
print(y.shape)









         









