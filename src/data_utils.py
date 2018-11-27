import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler

class BasketballDataset:
    def __init__(self, path, bin=False, min_max=False):
        df = pd.read_csv(path, index_col=0)

        # Use only recent stats and fill in missing values
        df = df[df.Year >= 2011]
        df = df.fillna(df.mean())

        # Get rid of percentage and advanced derived stats
        data_columns = [col for col in df.columns
                         if not (('%' in col)
                                 or (col in
                                     ['PER', 'Age', 'OWS', 'DWS', 'WS', 'WS/48', 'BPM', 'VORP', '3PAr', 'FTr']
                                    )
                                )]

        total_data = df.drop_duplicates(subset=['Year', 'Player'], keep='first')[data_columns]
        total_data = total_data[total_data.MP > 0]

        # Standardize on minute and mean/variance
        data_columns = ['FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA',
                        'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
        for c in data_columns:
            total_data[c] = total_data[c].values / total_data['MP'].values

        data = total_data.drop(['Year', 'Player', 'Pos', 'Tm', 'G', 'GS', 'MP'], axis=1)
        data_info = total_data[['Year', 'Player', 'Pos', 'Tm']]

        self.data_cols = data.columns
        self.info_cols = data_info.columns

        x_train, x_val, x_train_info, x_val_info = train_test_split(
            scaled_xs, x_info, test_size = 0.2
        )
        self.transformer = None
        self.set_transformer()
        self.x_train = x_train
        self.x_val = x_val
        self.x_train_info = x_train_info
        self.x_val_info = x_val_info

    def set_transformer(self, min_max=False, standardize=False):
        if min_max:
            self.transformer = MinMaxScaler().fit(x_train)
        elif standardize:
            self.transformer = StandardScaler().fit(x_train)

    def batch_train(self, batch_size):
        start = 0

        n = self.x_train.shape[0]
        while start < n:
            stop = start + batch_size
            batch = self.x_train[start:stop]

            if self.transformer:
                batch = self.transformer.transform(batch)

            yield batch

            start = stop
