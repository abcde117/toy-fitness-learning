import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import jax

from jax import numpy as jnp
from jax import random as jrnd

class WineNNDataManager:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

        # -------- numeric --------
        self.numeric_cols = [
            c for c in df.columns if c not in ["type"]
        ]
        X_raw = df[self.numeric_cols].to_numpy()

        self.scaler = StandardScaler()
        self.x = self.scaler.fit_transform(X_raw)     # (N, D)

        # -------- quality --------
        self.quality = df["quality"].to_numpy(dtype=float)  # (N,)

        # -------- type --------
        self.type_raw = df["type"].to_numpy()
        self.type_numeric, _ = pd.factorize(self.type_raw)

        enc = OneHotEncoder(sparse_output=False)
        type_oh = enc.fit_transform(self.type_raw[:, None])

        # -------- xo: rich view --------
        self.xo = np.concatenate(
            [self.x, type_oh], axis=1
        )

    def get_numpy(self):
        return {
            "x": self.x.copy(),
            "xo": self.xo.copy(),
            "quality": self.quality.copy(),
            "type": self.type_numeric.copy(),
        }
def build_type_graph(t):
    # t: (N,)
    return (t[:, None] == t[None, :]).astype(jnp.float32)

def get_batch(data, key, batch_size):
    N = data["x"].shape[0]
    idx = jax.random.choice(key, N, (batch_size,), replace=False)

    return {
        "x": data["x"][idx],
        "xo": data["xo"][idx],
        "quality": data["quality"][idx],
        "type": data["type"][idx],
    }