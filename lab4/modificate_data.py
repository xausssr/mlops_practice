import numpy as np
import pandas as pd


if __name__ == '__main__':
    df = pd.read_csv("data/moc_data.csv", index_col=0)
    idx_to_change = np.random.randint(0, len(df.columns))
    df[df.columns[idx_to_change]] = np.random.randn(1000)
    df.to_csv("data/moc_data.csv")
