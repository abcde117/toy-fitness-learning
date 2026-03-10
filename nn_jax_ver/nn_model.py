
import jaxtyping
from jaxtyping import Array,PyTree,Float,PRNGKeyArray
from einops import rearrange, einsum
import equinox as eqx
import jax

from jax import numpy as jnp
from jax import random as jrnd

from typing import Sequence, Callable, Optional,List,Dict




#encoding layer 

class naive_GCN(eqx.Module):

  lin:eqx.nn.Linear
  act:Callable #


  def __init__(self, key:PRNGKeyArray,in_channels,out_channels):


     self.lin=eqx.nn.Linear(in_channels,out_channels,key=key)
     self.act=jax.nn.leaky_relu

  def __call__(self,x:Float[Array,'n m'],g:Float[Array,'... n n'])->Float[Array,'... n d']:
    num_neighbours=g.sum(axis=-1, keepdims=True)
    node_feats =jax.vmap(self.lin)(x)
    node_feats_=g@node_feats
    node_feats = node_feats_ / num_neighbours


    return self.act( node_feats)

class naive_feature_head(eqx.Module):
 lin:eqx.nn.Linear
 act:Callable

 def __init__(self,key:PRNGKeyArray,in_channels,out_channels):

    self.lin=eqx.nn.Linear(in_channels,out_channels,key=key)
    self.act=jax.nn.gelu
 def __call__(self,x:Float[Array,'n m'])->Float[Array,'n d']:
     return self.act(jax.vmap(self.lin)(x))


class aggloremation_layer(eqx.Module):
   w:Float[Array,'c']
   act:Callable
   def __init__(self,key:PRNGKeyArray,in_channels):
    self.w=jrnd.normal(key,(in_channels))
    self.act=jax.nn.softmax

   def __call__(self,x:Float[Array,'c n m'])->Float[Array,'n m']:
      aw=self.act(self.w)
      agg=einsum(aw,x,'c , c n m ->n m')
      return agg


class enoding_layer(eqx.Module):
  ghead:naive_GCN
  fhead:naive_feature_head
  agg:aggloremation_layer

  def __init__(self,key:PRNGKeyArray,in_channels,out_channels,cdim):
    keys=jrnd.split(key,3)
    self.ghead=naive_GCN(keys[0],in_channels,in_channels)
    self.fhead=naive_feature_head(keys[1],in_channels,in_channels)
    self.agg=aggloremation_layer(keys[2],cdim)
  def __call__(self,x:Float[Array,'n m'],g:Float[Array,'n n'])->Float[Array,'n d']:
    gh=self.ghead(x,g)
    fh=self.fhead(x)
    return self.agg(jnp.concatenate([fh[None,:,:],gh],axis=0))



def build_mlp(
    key,
    sizes: Sequence[int],
    activation: Callable = jax.nn.gelu,
    output_activation: Optional[Callable] = None,
):
    keys = jrnd.split(key, len(sizes) - 1)
    layers = []

    for i in range(len(sizes) - 1):
        layers.append(
            eqx.nn.Linear(sizes[i], sizes[i + 1], key=keys[i])
        )

        if i < len(sizes) - 2:
            layers.append(activation)
        elif output_activation is not None:
            layers.append(output_activation)

    class MLP(eqx.Module):
        layers: tuple

        def __call__(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    return MLP(tuple(layers))



class navie_nn(eqx.Module):
    encoder: eqx.Module
    net: eqx.Module

    def __init__(
        self,
        key,
        dim: int,
        hidden_dims: List[int],
        out_dim: int,
        cdim: int = 2,
    ):
        k1, k2 = jrnd.split(key, 2)

        self.encoder = enoding_layer(
            key=k1,
            in_channels=dim,
            out_channels=hidden_dims[0],
            cdim=cdim,
        )

        mlp_sizes = [dim] + hidden_dims + [out_dim]
        self.net = build_mlp(
            key=k2,
            sizes=mlp_sizes,
            activation=jax.nn.gelu,
            output_activation=None,
        )

    def __call__(self, x, g):
        x = self.encoder(x, g)
        return jax.vmap(self.net)(x)



# decoding 
class naive_gat_like(eqx.Module):
  dk:int
  w:eqx.nn.Linear
  wv:eqx.nn.Linear
  act1:Callable
  act2:Callable
  a:Float[Array,'2dk']

  def __init__(self,key:PRNGKeyArray,dim,dk,dv):
     self.dk=dk
     keys=jrnd.split(key,3)
     self.w=eqx.nn.Linear(dim,dk,key=keys[0]) # Fixed: key must be passed as keyword argument
     self.wv=eqx.nn.Linear(dim,dv,key=keys[1]) # Fixed: key must be passed as keyword argument
     self.a=jrnd.normal(keys[2],(2*dk))
     self.act1=jax.nn.leaky_relu
     self.act2=jax.nn.softmax
  def __call__(self,x:Float[Array,'n m']) ->Float[Array,' n dv']:
    wx=jax.vmap(self.w)(x)
    a=(wx[:,None,:] *self.a[:self.dk]).sum(axis=-1)+(wx[None,:,:]*self.a[self.dk:]).sum(axis=-1)
    score=self.act1(a)
    score=self.act2(score)
    v=jax.vmap(self.wv)(x)
    return einsum(score,v,'... i j, ... j n ->... i n')


class naive_attn_like(eqx.Module):

  wqk:eqx.nn.Linear
  wv:eqx.nn.Linear
  act:Callable

  def __init__(self,key:PRNGKeyArray,dim,dk,dv,):
    keys=jrnd.split(key,2)
    self.wqk=eqx.nn.Linear(dim,dk,key=keys[0]) # Also fixed this for consistency, though not in error scope
    self.wv=eqx.nn.Linear(dim,dv,key=keys[1]) # Also fixed this for consistency
    self.act=jax.nn.softmax

  def __call__(self,x:Float[Array,'n m']) -> Float[Array,' n dv']:

     wqk=jax.vmap(self.wqk)(x)
     wv=jax.vmap(self.wv)(x)
     score=einsum(wqk,wqk,'... i m, ... j m  ->... i j')
     score=self.act(score)
     return einsum(score,wv,'... i j, ... j n ->... i n')

class naive_ffn(eqx.Module):

  proj:eqx.nn.Linear

  gru_cell:eqx.nn.GRUCell


  def __init__(self,key:PRNGKeyArray,dv,df):
    keys=jrnd.split(key,2)
    self.proj=eqx.nn.Linear(dv,df,key=keys[0]) # Also fixed this for consistency
    self.gru_cell=eqx.nn.GRUCell(dv,df,key=keys[1]) # Also fixed this for consistency

  def __call__ (self,x:Float[Array,'n m']) -> Float[Array,'n dv']:
   proj=jax.vmap(self.proj)(x)
   h=jnp.zeros_like(x)
   h=jax.vmap(self.gru_cell)(proj,h)
   return h


class naive_geo_layer(eqx.Module):

  depth:eqx.nn.Conv1d
  conv1x1:eqx.nn.Conv1d
  act:Callable

  proj:eqx.nn.Linear

  def __init__(self,key:PRNGKeyArray,dv,dout):
    keys=jrnd.split(key,3)
    self.depth = eqx.nn.Conv1d(in_channels=dv,out_channels=dv, kernel_size=3,
    padding=1,groups=dv,key=keys[0])

    self.conv1x1 = eqx.nn.Conv1d(in_channels=dv,out_channels=dv,kernel_size=1,key=keys[1])
    self.act=jax.nn.gelu

    self.proj=eqx.nn.Linear(dv,dout,key=keys[2])

  def __call__ (self,x:Float[Array,'n m']) -> Float[Array,'n dv']:
     x=rearrange(x,'... n m -> ... m n')
     depth=self.depth(x)
     depth=self.act(depth)
     conv1x1=self.conv1x1(depth)

     conv1x1=rearrange(conv1x1,'... m n -> ... n m')
     conv1x1=self.act(conv1x1)
     res=jax.vmap(self.proj)(conv1x1)
     return res


class decoding(eqx.Module):

  proj:eqx.nn.Linear
  gat:naive_gat_like
  attn:naive_attn_like
  ffn:naive_ffn
  geo:naive_geo_layer
  ln:eqx.nn.LayerNorm
  ln2:eqx.nn.LayerNorm
  ln3:eqx.nn.LayerNorm
  head:eqx.nn.Linear


  def __init__(self, key:PRNGKeyArray,dg,dim,dk,dout):
     keys=jrnd.split(key,6)
     self.proj=eqx.nn.Linear(dim,dk,key=keys[0])
     self.gat=naive_gat_like(keys[1],dg,dk,dk)
     self.attn=naive_attn_like(keys[2],dk,dk,dk)
     self.ffn=naive_ffn(keys[3],dk,dk)
     self.geo=naive_geo_layer(keys[4],dk,dk)
     self.ln=eqx.nn.LayerNorm(dk)
     self.ln2=eqx.nn.LayerNorm(dk)
     self.ln3=eqx.nn.LayerNorm(dk)
     self.head=eqx.nn.Linear(dk,dout,key=keys[5])

  def __call__(self,x:Float[Array,'... n m'])->Float[Array,'... n dout']:
    x=jax.vmap(self.proj)(x)
    m=jax.vmap(self.ln)(x)
    mm=x@x.T
    gat=self.gat(mm)
    attn=self.attn(m)
    h=gat+attn+x
    ffn=self.ffn(jax.vmap(self.ln2)(h))+h
    geo=self.geo(jax.vmap(self.ln3)(ffn))+ffn
    out=jax.vmap(self.head)(geo)
    return out



#print


def count_params_and_mib(model) -> Dict[str, float]:

    leaves = eqx.filter(model, eqx.is_array)

    arrays = jax.tree_util.tree_leaves(leaves)

    total_params = sum(a.size for a in arrays)
    total_bytes = sum(a.size * a.dtype.itemsize for a in arrays)

    return {
        "params": total_params,
        "mib": total_bytes / (1024 ** 2),
    }


def print_model_stats(model):
    stats = count_params_and_mib(model)

    print("=== Model ===")
    print(f"Parameters : {stats['params']:,}")
    print(f"Param size : {stats['mib']:.3f} MiB")

