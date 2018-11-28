import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

class BasketballDataset:
    def __init__(self, path,
            bin=False,
            min_minutes_played=82,
            test_split=0.2,
            per_minute=True,
            **kwargs
        ):
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
        total_data = total_data[total_data.MP > min_minutes_played]

        # Standardize on per minute
        if per_minute:
            data_columns = ['FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA',
                            'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
            for c in data_columns:
                total_data[c] = total_data[c].values / total_data['MP'].values

        data = total_data.drop(['Year', 'Player', 'Pos', 'Tm', 'G', 'GS', 'MP'], axis=1)
        data_info = total_data[['Year', 'Player', 'Pos', 'Tm']]

        self.data_cols = data.columns
        self.info_cols = data_info.columns

        x_train, x_test, x_train_info, x_test_info = train_test_split(
            data, data_info, test_size = test_split
        )
        self.transformer = None
        self._x_train = x_train
        self._x_test = x_test
        self.x_train_info = x_train_info
        self.x_test_info = x_test_info
        self.x_train_copy = np.copy(x_train)

        self.reset_transform(**kwargs)

    def reset_transform(self, **kwargs):
        self.set_transformer(**kwargs)
        self.apply_transform()

    def set_transformer(self,
            min_max=False,
            standardize=False,
            nonzero=True
        ):
        x_train = self._x_train
        if min_max:
            if nonzero:
                self.transformer = MinMaxScaler().fit(x_train)

                def transform(x, **kwargs):
                    return self.transformer.transform(x, **kwargs) + 1e-4

                self.transform = transform
        elif standardize:
            self.transformer = StandardScaler().fit(x_train)
            self.transform = self.transformer.transform
        else:
            self.transformer = None
            self.transform = None

    def apply_transform(self):
        if self.transform:
            self.x_train = self.transform(self._x_train)
            self.x_test = self.transform(self._x_test)
        else:
            self.x_train = self._x_train
            self.x_test = self._x_test

    def add_players_to_test(self, *names):
        x_test = self.x_test
        x_test_info = self.x_test_info

        mask = np.zeros(self.x_train.shape[0], dtype=np.bool)
        for name in names:
            mask |= self.x_train_info.Player == name
        print(self.x_train[mask].shape)
        x_test = np.vstack([x_test, self.x_train[mask]])
        x_test_info = pd.concat([
            x_test_info,
            self.x_train_info.loc[mask]
        ], axis=0)

        self.x_test = x_test
        self.x_test_info = x_test_info

    def batch_train(self, batch_size):
        start = 0
        x_train = self.x_train
        n = x_train.shape[0]
        np.random.shuffle(x_train)

        while start < n:
            stop = start + batch_size
            yield x_train[start:stop]
            start = stop
