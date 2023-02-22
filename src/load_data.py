import copy
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset


class DBEncoder:
    """Encoder used for data discretization and binarization."""

    def __init__(self, f_df, discrete=False, y_one_hot=True, drop='first'):
        self.f_df = f_df
        self.discrete = discrete
        self.y_one_hot = y_one_hot
        self.label_enc = preprocessing.OneHotEncoder(categories='auto') if y_one_hot else preprocessing.LabelEncoder()
        self.feature_enc = preprocessing.OneHotEncoder(categories='auto', drop=drop)
        self.imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.x_fname = None
        self.y_fname = None
        self.discrete_flen = None
        self.continuous_flen = None
        self.mean = None
        self.std = None

    def split_data(self, x_df):
        discrete_data = x_df[self.f_df.loc[self.f_df[1] == 'discrete', 0]]
        continuous_data = x_df[self.f_df.loc[self.f_df[1] == 'continuous', 0]]
        if not continuous_data.empty:
            continuous_data = continuous_data.replace(to_replace=r'.*\?.*', value=np.nan, regex=True)
            continuous_data = continuous_data.astype(np.float32)
        return discrete_data, continuous_data

    def fit(self, x_df, y_df):
        x_df = x_df.reset_index(drop=True)
        y_df = y_df.reset_index(drop=True)
        discrete_data, continuous_data = self.split_data(x_df)
        self.label_enc.fit(y_df)
        self.y_fname = list(self.label_enc.get_feature_names_out(y_df.columns)) if self.y_one_hot else y_df.columns

        if not continuous_data.empty:
            # Use mean as missing value for continuous columns if do not discretize them.
            self.imp.fit(continuous_data.values)
        if not discrete_data.empty:
            # One-hot encoding
            self.feature_enc.fit(discrete_data)
            feature_names = discrete_data.columns
            # print(f"feature names du db enc : {feature_names}")
            self.x_fname = list(self.feature_enc.get_feature_names_out(feature_names))
            self.discrete_flen = len(self.x_fname)
            if not self.discrete:
                self.x_fname.extend(continuous_data.columns)
        else:
            self.x_fname = continuous_data.columns
            self.discrete_flen = 0
        self.continuous_flen = continuous_data.shape[1]

    def transform(self, x_df, y_df, normalized=False, keep_stat=False):
        x_df = x_df.reset_index(drop=True)
        y_df = y_df.reset_index(drop=True)
        discrete_data, continuous_data = self.split_data(x_df)
        # Encode string value to int index.
        y = self.label_enc.transform(y_df.values.reshape(-1, 1))
        if self.y_one_hot:
            y = y.toarray()

        if not continuous_data.empty:
            # Use mean as missing value for continuous columns if we do not discretize them.
            continuous_data = pd.DataFrame(self.imp.transform(continuous_data.values),
                                           columns=continuous_data.columns)
            if normalized:
                if keep_stat:
                    self.mean = continuous_data.mean()
                    self.std = continuous_data.std()
                continuous_data = (continuous_data - self.mean) / self.std
                # print(self.mean, self.std)
        if not discrete_data.empty:
            # One-hot encoding
            discrete_data = self.feature_enc.transform(discrete_data)
            if not self.discrete:
                x_df = pd.concat([pd.DataFrame(discrete_data.toarray()), continuous_data], axis=1)
            else:
                x_df = pd.DataFrame(discrete_data.toarray())
        else:
            x_df = continuous_data

        return x_df.values, y

    def change_features_order(self, x, mappings):
        """Change the order of the features to allow better patches"""
        x_f = None
        x_fnameici = []

        for mappingindex in list(mappings.keys()):
            mapping = mappings[mappingindex]
            x_fnameici += list(mapping.values())
            permutation = np.array(list(mapping.keys())).astype(int)
            x = x[:, permutation]
            if x_f is None:
                x_f = copy.deepcopy(x)  # _permut)

            else:
                # x_f = np.concatenate((x_f, x_permut), axis=1)
                x_f = np.concatenate((x_f, x), axis=1)
                # print(x_f.shape, x.shape)
            # print(permutation)
        self.x_fname = x_fnameici
        return x_f


def get_bn_thresh_from_file(bn_path):
    with open(bn_path, 'r') as file:
        bn_thresh = file.read()
    return float(bn_thresh)


def bn2continuous(bn_thresh, std, mu):
    return mu + std * bn_thresh


def continuous2bn(cont_thresh, std, mu):
    return (cont_thresh - mu) / std


def read_info(info_path):
    with open(info_path) as f:
        f_list = []
        for line in f:
            tokens = line.strip().split()
            f_list.append(tokens)
    return f_list[:-1], int(f_list[-1][-1])


def read_csv(data_path, info_path, shuffle=False):
    d = pd.read_csv(data_path, header=None)
    if shuffle:
        d = d.sample(frac=1, random_state=0).reset_index(drop=True)
    f_list, label_pos = read_info(info_path)
    f_df = pd.DataFrame(f_list)
    d.columns = f_df.iloc[:, 0]
    y_df = d.iloc[:, [label_pos]]
    x_df = d.drop(d.columns[label_pos], axis=1)
    f_df = f_df.drop(f_df.index[label_pos])
    return x_df, y_df, f_df, label_pos


def get_data_loader(dataset, data_dir, load_permut=False, output_model=None, random_permut=False, repeat_permut=0, k=1,
                    seed=0, mappings_train=None, as_numpy=False, bn_path=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    data_path = os.path.join(data_dir, dataset + '.data')
    info_path = os.path.join(data_dir, dataset + '.info')
    x_df, y_df, f_df, label_pos = read_csv(data_path, info_path, shuffle=True)

    db_enc = DBEncoder(f_df, discrete=False)
    db_enc.fit(x_df, y_df)

    if load_permut:
        print()
        print("MAPPING dIR : ", load_permut)
        print("Loading existing mapping")
        print(load_permut)
        with open(os.path.join(load_permut), 'r') as jfile:
            mappings = json.load(jfile)
    else:
        # print(db_enc.x_fname)
        mappings = {"0": {}}
        for i, ii in enumerate(db_enc.x_fname):
            mappings['0'][i] = ii
            print(i, ii)

    print()

    if random_permut:
        if mappings_train is not None:
            mappings = mappings_train
        else:
            print("Using Random mapping")
            features = db_enc.x_fname
            normal_order = {i: feat for i, feat in enumerate(features)}
            idx = list(range(len(features)))
            mappings = {}
            for iterici in range(repeat_permut):
                random.shuffle(idx)
                mappings[iterici] = {i: normal_order[i] for i in idx}
            with open(os.path.join(output_model, dataset + "_mapping.json"), 'w') as jfile:
                json.dump(mappings, jfile)
            print(output_model, dataset + "_mappings.json")
    print(mappings)
    x, y = db_enc.transform(x_df, y_df, normalized=True, keep_stat=True)
    x = db_enc.change_features_order(x, mappings)

    # print(db_enc.x_fname)
    # for i, ii in enumerate(db_enc.x_fname):
    #    print(i, ii)
    # print(ok)
    if seed is not None:
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=0)

    train_index, test_index = list(kf.split(x_df))[k]
    # print(test_index)
    # with open(os.path.join(dATA_dIR, "test_indexes.npy"), 'rb') as f:
    #    test_index = np.load(f)
    # print(test_index)
    x_train = x[train_index]
    y_train = y[train_index]
    x_test = x[test_index]
    y_test = y[test_index]
    # print(x_test[0, :])

    # print(x_test[0,:])

    if bn_path is not None:
        thr_bn = get_bn_thresh_from_file(bn_path)
        cont_idx = []
        all_feat = []
        for i in mappings.values():
            all_feat += list(i.values()).copy()

        for i, feat in enumerate(all_feat):
            try:
                if '_' in feat:
                    continue
                int(feat)
                # print(feat)
                cont_idx.append(i)
            except ValueError:
                continue
        if cont_idx:
            x_train[:, cont_idx] = (x_train[:, cont_idx] > thr_bn).astype(np.float32)
            x_test[:, cont_idx] = (x_test[:, cont_idx] > thr_bn).astype(np.float32)
            x_test = (x_test > thr_bn).astype(np.float32)

    if as_numpy:
        train_set = [x_train.astype(np.float32), y_train.astype(np.float32)]
        test_set = [x_test.astype(np.float32), y_test.astype(np.float32)]
    else:
        train_set = TensorDataset(torch.tensor(x_train.astype(np.float32)), torch.tensor(y_train.astype(np.float32)))
        test_set = TensorDataset(torch.tensor(x_test.astype(np.float32)), torch.tensor(y_test.astype(np.float32)))

    return train_set, test_set, mappings
