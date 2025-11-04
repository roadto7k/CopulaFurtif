import os
import pandas as pd

def load_all_prices(DATA_PATH):
    data = {}
    for file in os.listdir(DATA_PATH):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(DATA_PATH, file), index_col=0, parse_dates=True)
            if 'close' in df.columns:
                data[file[:-4]] = df['close']
            else:
                lower = {c.lower(): c for c in df.columns}
                if 'close' in lower:
                    data[file[:-4]] = df[lower['close']]
    return pd.DataFrame(data).sort_index()