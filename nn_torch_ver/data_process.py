
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import Dataset, DataLoader



class WineNNDataManager:
    def __init__(self, df):
        self.df = df.copy()

        # -------- numeric --------
        self.numeric_cols = [
            c for c in df.columns if c not in [ "type"]
        ]
        X_raw = df[self.numeric_cols].values

        self.scaler = StandardScaler()
        self.x = self.scaler.fit_transform(X_raw)   # (N, D)

        # -------- optional: include quality --------
        self.quality = df["quality"].values.astype(float)  # (N,)

        # -------- type --------
        self.type_raw = df["type"].values
        # Convert string type labels to numerical integers
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
class WineNNDataset(Dataset):
    def __init__(self, data):
        self.x = data["x"]
        self.xo = data["xo"]
        self.quality = data["quality"]
        self.type = data["type"]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return {
            "x": self.x[idx],
            "xo": self.xo[idx],
            "quality": self.quality[idx],
            "type": self.type[idx],
        }
def build_type_graph(t):
    return (t[:, None] == t[None, :]).float()