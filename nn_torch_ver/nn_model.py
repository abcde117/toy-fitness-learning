


from einops import einsum ,rearrange, repeat
import jaxtyping
from  jaxtyping  import Array,Float,Int,Bool
import torch

from torch import nn
import torch.nn.functional as F

#encoding models
class navie_GCN(nn.Module):
  def __init__(self,in_channels,out_channels):
    super().__init__()
    self.lin=nn.Linear(in_channels,out_channels)
    self.act=nn.LeakyReLU()

  def forward(self,x:Float[Array,'n m'],g:Float[Array,'n n']):
      num_neighbours = g.sum(axis=-1, keepdims=True)
      node_feats =self.lin(x)
      node_feats_=g@node_feats
      node_feats = node_feats_ / num_neighbours

      return self.act(node_feats)

class navie_fearue_head(nn.Module):
    def __init__(self,in_channels,out_channels):
      super().__init__()
      self.lin=nn.Linear(in_channels,out_channels)
      self.act=nn.GELU()
    def forward(self,x:Float[Array,'n m']):
      return self.act(self.lin(x))

class aggloremation_layer(nn.Module):
   def __init__(self,in_channels):
     super().__init__()
     self.w=nn.Parameter(torch.randn(in_channels))
     self.act=nn.Softmax(dim=-1)
   def forward(self,x:Float[Array,'c n m']):
      aw=self.act(self.w)
      agg=einsum(aw,x,'c , c n m ->n m')
      return agg

class enoding_layer(nn.Module):
  def __init__(self,in_channels,out_channels,cdim):
    super().__init__()
    self.ghead=navie_GCN(in_channels,in_channels)
    self.fhead=navie_fearue_head(in_channels,in_channels)
    self.agg=aggloremation_layer(cdim)

  def forward(self,x:Float[Array,'n m'],g:Float[Array,'n n']):
    gh=self.ghead(x,g)
    fh=self.fhead(x)
    context=self.agg(torch.concat((fh[None,:,:],gh),dim=0))
    return context

def build_mlp(
    sizes,
    activation=nn.GELU,
    output_activation=None,
    dropout=0.0,
):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))

        if i < len(sizes) - 2:
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        else:
            if output_activation is not None:
                layers.append(output_activation())

    return nn.Sequential(*layers)


class navie_nn(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dims: list[int],
        out_dim: int,
        cdim: int = 2,
    ):
        super().__init__()

        self.encoder = enoding_layer(
            in_channels=dim,
            out_channels=hidden_dims[0],
            cdim=cdim,
        )

        mlp_sizes = [dim] + hidden_dims + [out_dim]
        self.net = build_mlp(
            mlp_sizes,
            activation=nn.GELU,
            output_activation=None,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, aggloremation_layer):
                nn.init.constant_(m.w, 0.0)

    def forward(self, x, g):
        x = self.encoder(x, g)
        return self.net(x)
    
    
    
# decoding models

class naive_gat_like(nn.Module):
  def __init__(self,dim,dk,dv):
     super().__init__()
     self.dk=dk
     self.w=nn.Linear(dim,dk)
     self.wv=nn.Linear(dim,dv)
     self.a=nn.Parameter(torch.randn(2*dk))
     self.act1=nn.LeakyReLU()
     self.act2=nn.Softmax(dim=-1)
  def forward(self,x:Float[Array,'... n n']):
     wx=self.w(x)



     a=(wx[:,None,:] *self.a[:self.dk]).sum(dim=-1)+(wx[None,:,:]*self.a[self.dk:]).sum(dim=-1)
     score=self.act1(a)
     score=self.act2(score)
     v=self.wv(x)
     return einsum(score,v,'... i j, ... j n ->... i n')

class naive_attn_like(nn.Module):
  def __init__(self,dim,dk,dv,):
     super().__init__()
     self.wqk=nn.Linear(dim,dk)
     self.wv=nn.Linear(dim,dv)
     self.act=nn.Softmax(dim=-1)


  def forward(self,x:Float[Array,'... n m']):
     wqk=self.wqk(x)
     wv=self.wv(x)
     score=einsum(wqk,wqk,'... i m, ... j m  ->... i j')
     score=self.act(score)
     return einsum(score,wv,'... i j, ... j n ->... i n')
class naive_ffn(nn.Module):
  def __init__(self,dv,df):
     super().__init__()

     self.proj=nn.Linear(dv,df)
     self.ffn=nn.GRUCell(df,df)

  def forward(self,x:Float[Array,'... n m']):
     proj=self.proj(x)
     h=torch.zeros_like(proj)
     h=self.ffn(proj,h)

     return h


class naive_geo_layer(nn.Module):
    def __init__(self,dv,dout):
     super().__init__()
     self.depth=nn.Conv1d(dv,dv,kernel_size=3,padding=1,groups=dv)
     self.act1=nn.GELU()
     self.conv1x1=nn.Conv1d(dv,dv,kernel_size=1)
     self.proj=nn.Linear(dv,dout)
     self.act2=nn.GELU()


    def forward(self,x:Float[Array,'... n m']):
     x=rearrange(x,'... n m -> ... m n')
     depth=self.depth(x)
     depth=self.act1(depth)
     conv1x1=self.conv1x1(depth)

     conv1x1=rearrange(conv1x1,'... m n -> ... n m')
     conv1x1=self.act2(conv1x1)
     res=self.proj(conv1x1)
     return res



class decoding(nn.Module):
  def __init__(self,dg,dim,dk,dout):
     super().__init__()
     self.proj=nn.Linear(dim,dk)
     self.gat=naive_gat_like(dg,dk,dk)
     self.attn=naive_attn_like(dk,dk,dk)
     self.ffn=naive_ffn(dk,dk)
     self.geo=naive_geo_layer(dk,dk)
     self.ln1=nn.LayerNorm(dk)
     self.ln2=nn.LayerNorm(dk)
     self.ln3=nn.LayerNorm(dk)
     self.head=nn.Linear(dk,dout)

  def forward(self,x:Float[Array,'... n m']):
     x=self.proj(x)
     m=self.ln1(x)
     mm=x@x.T
     gat=self.gat(mm)
     attn=self.attn(m)
     h=gat+attn+x


     ffn=self.ffn(self.ln2(h))+h

     geo=self.geo(self.ln3(ffn))+ffn

     out=self.head(geo)


     return out
 
# print params
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params
    }
def print_model_mib(model, name="Model", include_buffers=True):
    """
    Print model parameter count and size in MiB.
    """
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    buffer_bytes = 0
    if include_buffers:
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

    total_bytes = param_bytes + buffer_bytes
    total_mib = total_bytes / (1024 ** 2)

    param_mib = param_bytes / (1024 ** 2)
    buffer_mib = buffer_bytes / (1024 ** 2)

    print(f"=== {name} ===")
    print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Param size : {param_mib:.3f} MiB")
    if include_buffers:
        print(f"Buffer size: {buffer_mib:.3f} MiB")
    print(f"Total size : {total_mib:.3f} MiB")