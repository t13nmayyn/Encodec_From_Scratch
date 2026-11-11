import torch 
from torch import nn 
import torch.nn.functional as F 


class CodeBook(nn.Module):
    def __init__(self,codebook_size:int,latent_dim:int,decay:float,dead_code_threshold:float,epsilon:float=1e-5):
        super().__init__()
        self.latent_dim=latent_dim
        self.codebook_size=codebook_size
        self.decay=decay
        self.epsilon=epsilon
        self.dead_code_threshold=dead_code_threshold

        self.embedding=nn.Embedding(self.codebook_size,self.latent_dim)
        self.embedding.requires_grad_(False)
        self.embedding.weight.data.uniform_(-1/self.latent_dim,1/self.latent_dim)
        """ keeps the tracks of the ema weights updates and the statstics count of the usgae of the codebook vector"""
        self.register_buffer('em_cluster_size',torch.zeros(codebook_size))
        self.register_buffer("ema_emb",torch.zeros(codebook_size,latent_dim))

    def forward(self,z:torch.Tensor):
        flat_z=z.reshape(-1,self.latent_dim)
        Z2=torch.sum(flat_z**2,dim=1,keepdim=True)
        E2=torch.sum(self.embedding.weight**2,dim=1,keepdim=False)
        ZE=torch.matmul(flat_z,self.embedding.weight.transpose(-1,-2))
        distances=(Z2+E2-2*ZE)
        # making the encoding on the codebook vector 0&1 
        encoding_indices=torch.argmin(distances,dim=-1)
        z_q=self.embedding(encoding_indices)
        encoding=F.one_hot(encoding_indices,self.codebook_size).type_as(flat_z)
        z_q=z_q.view(z.shape)

        if self.training:
            self.update_codebook(flat_z,encoding)
            self.dead_code_replace(flat_z)

        # commitment loss 
        commitment_loss=F.mse_loss(z,z_q.detach())
        # codebok loss 
        #codebook_loss=F.mse_loss(z_q,z.detach())
        # total_loss
        

        # straight through gradients estimation
        z_q=z+(z_q-z).detach()

        
        avg_probs = torch.mean(encoding, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q, commitment_loss, perplexity,encoding_indices.reshape(z.shape[:-1])


    @torch.no_grad()
    def update_codebook(self,flat_z,encoding:torch.Tensor):
        cluster_size=torch.sum(encoding,dim=0) # (1,K) carryinng the sums of all entires to codebooks 
        # we want selected vector now  
        dw=torch.matmul(encoding.t(),flat_z)
        self.em_cluster_size=self.decay*self.em_cluster_size+(1-self.decay)*cluster_size
        self.ema_emb=self.decay*self.ema_emb+(1-self.decay)*dw

        n=torch.sum(self.em_cluster_size)
        cluster_size=(self.em_cluster_size+self.epsilon)/(n+self.codebook_size*self.epsilon)*n


        normalized_cluster_size=cluster_size.unsqueeze(1)
        self.embedding.weight.data=self.ema_emb/normalized_cluster_size

    def get_codebok_entry(self,indices:torch.Tensor):
        """ retrieving the codebook vector"""
        return self.embedding(indices)

    @torch.no_grad()
    def dead_code_replace(self,flat_z):
        dead_code=self.em_cluster_size<self.dead_code_threshold
        num_deads=int(dead_code.sum())
        if num_deads==0:
            return
        N=flat_z.shape[0]
        if N>=num_deads:
            random_indices=torch.randperm(flat_z.shape[0])[:num_deads]
        else:
            random_indices=torch.randint(0, N, (num_deads,), device=flat_z.device)
        replacement_vector=flat_z[random_indices]
        self.embedding.weight[dead_code]=replacement_vector
        grace = 2 * self.dead_code_threshold
        self.ema_emb[dead_code] = replacement_vector * grace
        self.em_cluster_size[dead_code] = grace        
        print(f' {num_deads} Vector Replaced...')


    def get_distance_matrix(self,z:torch.Tensor):
        flat_z=z.view(-1,self.latent_dim)

        distances=(torch.sum(flat_z**2,dim=1)+torch.sum(self.embedding.weight**2,dim=1)-2*torch.matmul(flat_z,self.embedding.weight.t()))

        return distances





x=torch.randn(2,6,128)
print("input shape : ",x.shape)




 


class ResidualVectorQuantization(nn.Module):
    def __init__(self,codebook_size:int,latent_dim:int,n_codebook:int,decay:float,dead_code_threshold:float):
        super().__init__()
        self.latent_dim=latent_dim
        self.codebook_size=codebook_size
        self.decay=decay
        self.codebooks=nn.ModuleList([CodeBook(codebook_size,latent_dim,decay,dead_code_threshold) for _ in range(n_codebook) ])
        self.n_codebook=n_codebook
    def forward(self,z):
        all_indices=[]
        residual=z
        total_loss=0
        quantized_out=0
        prpl=[]
        for idx,codebook in enumerate(self.codebooks):

            z_q_i,loss_i,perplexity_i,indices_i=codebook(residual)
            all_indices.append(indices_i)
            total_loss+=loss_i
            residual=residual-z_q_i.detach()
            quantized_out=quantized_out+z_q_i
            prpl.append(perplexity_i)
        perplexity=torch.stack(prpl)
        indexs=torch.stack(all_indices,dim=1)

        return quantized_out,residual,perplexity,indexs,total_loss


    def decode(self,indices):
        out=0
        for i  in range(self.n_codebook):
            idx=indices[:,i,:]
            codebook=self.codebooks[i]
            out=out+codebook.get_codebok_entry(idx)
        return out





rvq=ResidualVectorQuantization(4,128,4,0.99,1.28)
rvq.train()

z_q,r_v,p,i,l=rvq(x)

print(rvq.decode(i))

print(z_q.shape)
print(i.shape)

print(r_v.shape)
print(p)
print(l)
