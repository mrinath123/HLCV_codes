import numpy as np
import torch 
import torch.nn as nn
import einops
from einops import rearrange, reduce, repeat

#image_size = 224 x 224
model_dimension = 64
# patch size = 28 x 28
# number of patches = (224 / 28 )^2 = 64
number_of_patches = 64

# for VIT
# input -> bs x h x w x 3
# number of classes -> 10
# Patchify (bs x h x w x 3) -> (bs x no. of patches x patch size x patch size x channels)
# LinearProj (bs x no. of patches x patch size x patch size x channels) -> (bs x no. of patches x model_dimension )


class Patchify(nn.Module):
    def __init__(self , p = 8) -> None:
        super().__init__()
        self.p = p

    def forward(self , x ):
        x = rearrange(x , 'b (h p1) (w p2)  c -> b (p1 p2) h w c', p1=self.p, p2=self.p)
        return x

class LinearProj(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.lin = nn.Linear(28*28*3 , 64)

    def forward(self,x):
        x = rearrange(x , 'b p p1 p2 c -> b p (p1 p2 c)' )
        x = self.lin(x)
        return x

# Next step add a CLS token in the start and add position learnable embeddings , pass it to VIT block

class VITblock(nn.Module):
    def __init__(self,num_heads = 8) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(model_dimension)
        self.norm2 = nn.LayerNorm(model_dimension)
        self.num_heads = num_heads
        self.mlp = nn.Sequential(
            nn.Linear(model_dimension , model_dimension),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(model_dimension , model_dimension),
        )
        self.MHSA = nn.MultiheadAttention(embed_dim = model_dimension, num_heads = self.num_heads)

    def forward(self,x):
        # x is of shape (bs x no. of patches + 1 x model_dimension )
        x1 = self.norm1(x)
        x2,_ = self.MHSA(x1,x1,x1)
        x  = x2 + x

        x3 = self.norm2(x)
        x4 = self.mlp(x3)
        x  = x4 + x

        return x

class ViT(nn.Module):
    def __init__(self,enc_layers = 3) -> None:
        super().__init__()

        self.n = enc_layers
        
        self.positional_embedding = nn.Parameter(torch.randn(number_of_patches + 1, model_dimension))
        self.class_token = nn.Parameter(torch.randn(1 , model_dimension))

        self.patch = Patchify()
        self.lin = LinearProj()
        self.layers = nn.ModuleList(
            [
                VITblock() for _ in range(self.n)
            ]
        )

        self.out = nn.Linear(model_dimension , 10) #output_classes = 10


    def forward(self,x):
        # x of shape (B , H , W , 3)
        x = self.patch(x) # (B , P , P1 ,P2 , 3)
        x = self.lin(x) # (B , P , model_dimension)
        B,_,_ = x.shape

        c = repeat(self.class_token , 'a b -> r a b', r = B)

        x = torch.cat(( c , x), dim = 1) # (B , P +1 , mod_dim)

        x = x + self.positional_embedding # (B , P +1 , mod_dim)

        for l in self.layers:
            x = l(x)

        x = x[: , 0] # CLS token only used for classification
        x = self.out(x)

        return x
    
if __name__ == "__main__":
    i1 = torch.randn(8, 224, 224 , 3)
    vit = ViT()
    op = vit(i1)
    print(op.shape)






        



        

    


    









    
