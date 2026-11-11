import torch
from torch import nn, poisson_nll_loss 
import torch.nn.functional as F 





def test_kmeans():
    import matplotlib.pyplot as plt 
    true_centers=torch.tensor([[-2,-2],[0,4],[5,3]])
    n_per_clusters=200
    dim=2
    samples=[]
    for c in true_centers:
        points=c+torch.randn(n_per_clusters,dim)
        samples.append(points)
    samples=torch.stack(samples)
    plt.scatter(samples[:,0],samples[:,1])
    plt.show()




test_kmeans()



