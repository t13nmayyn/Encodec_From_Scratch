
import torch 
from torch import nn
import torch.nn.functional as F 


class CodeBook(nn.Module):
    def __init__(self,codebook_size:int,latent_dim:int,alpha:float):
        super().__init__()
        self.alpha=alpha
        self.codebook=nn.Embedding(num_embeddings=codebook_size,embedding_dim=latent_dim)
        nn.init.uniform_(self.codebook.weight,-1/latent_dim,1/latent_dim)
        self.latent_dim=latent_dim
        self.codebook.weight.requires_grad=False
    
    


    def forward(self,z:torch.Tensor):
        #z(batch,seq_len or say time ,latent_dim)
        flat_z=z.reshape(-1,self.latent_dim)
        #batchx time/seq_len ,embedding_dim maaking it similar to codebook dim 
        Z2=torch.sum(flat_z**2,dim=1,keepdim=True)
        E2=torch.sum(self.codebook.weight**2,dim=1,keepdim=False)
        ZE=torch.matmul(flat_z,self.codebook.weight.transpose(-1,-2))
        distances=(Z2+E2-2*ZE)
        min_dist=torch.argmin(distances,dim=-1)
        z_q=self.codebook(min_dist)
        z_q=z_q.reshape(z.shape)
        min_dist=min_dist.reshape(z.shape[0],z.shape[1])
        residual_vector=z-z_q.detach()      
        codebook_loss=F.mse_loss(z.detach(),z_q)
        
        commit_loss=F.mse_loss(z,z_q.detach())
        z_q=z+(z_q-z).detach()
        # codebook vector update
        z_q=self.alpha*z_q+(1-self.alpha)*z
        return residual_vector,z_q,min_dist,commit_loss,codebook_loss

