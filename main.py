import os
import warnings

import numpy as np
from torch.utils.data import TensorDataset

from config.config import Config
from src.load_block import load_table_as_np_inference, get_lookup_tables
from src.load_data import get_data_loader

warnings.filterwarnings("ignore")


def pos_conv1d(shape: tuple, ksize: int = 3, padding: int = 0, stride_conv: int = 2) -> np.array:
    """Return the indices used in the 1D convolution
    @rtype: array
    @param shape: input shape
    @param ksize: kernel size
    @param padding: padding
    @param stride_conv: stride
    @return: the output adter unfolding
    """

    arr = np.array(list(range(shape[0])))

    out = np.zeros((
        (arr.shape[0] - ksize + 2 * padding) // stride_conv + 1,
        ksize)
    )
    shape = out.shape

    for i in range(0, shape[0]):
        sub = arr[i * stride_conv:i * stride_conv + ksize]
        v = sub.flatten()
        out[i] = v

    return out.astype(int)


def bitstointfast(bits: np.array) -> np.array:
    """
    Fast transformation from n bit to int on n bits
    @rtype: array
    @param bits: binary array of shape BxnbitxHxW
    @return: binary array of shape Bx1xHxW
    """
    if len(bits.shape) == 3:
        _, m, n = bits.shape  # number of columns is needed, not bits.size
    else:
        m, n = bits.shape
    a = 2 ** np.arange(n)[::-1]  # -1 reverses array of powers of 2 of same length as bits
    return bits @ a  # this matmult is the key line of code


def prepare_dataset(ksize: int, nfilters: int, pad: int, stride: int, data_dir: str, dataset: str,
                    model_path: str, mappings_path: dict, k_fold: int) -> (np.array, np.array, TensorDataset,
                                                                           np.array, np.array):
    """
    Load all the variables needed to infer: Truth tables, datasets, LR

    @rtype: tuple of array and dataset
    @param ksize: kernel size
    @param nfilters: number of filter after 1 layer
    @param pad: padding
    @param stride: stride
    @param data_dir: data direction
    @param dataset: name of the dataset
    @param model_path: path to get the LR and the bias
    @param mappings_path: dict that maps the input to the output in case multi channel input
    @param k_fold: number of folding
    @return: block, whole_test_set_preproc, test_set, w_lr, b_lr
    """
    op = 'dnf'
    path = os.path.join(model_path, 'thr_[0.0]/avec_DC_logic/Filtrage_0')
    bn_path = os.path.join(model_path, 'bn_thresh.txt')
    w_lr = np.load(os.path.join(model_path, 'lr_matrix.npy'))
    b_lr = np.load(os.path.join(model_path, 'b_matrix.npy'))[0]
    train_set, test_set, mappings = get_data_loader(dataset, data_dir, load_permut=mappings_path, as_numpy=True,
                                                    bn_path=bn_path, k=k_fold)
    x = test_set[0][0, :]
    shape = x.shape
    idx = pos_conv1d(shape, ksize, pad, stride)
    unfold_test_set = test_set[0][:, idx]
    whole_test_set_preproc = bitstointfast(unfold_test_set)
    whole_test_set_preproc = np.tile(whole_test_set_preproc, nfilters).astype(
        int)
    if dataset == "breast_cancer":
        block = get_lookup_tables(path, [unfold_test_set.shape[-2]], [ksize], op, num_blocks=1, exceptions=[])
        block = np.array(block[0]).astype(int)
    else:
        block = load_table_as_np_inference(path, ksize, op, [])
        block = np.array(block)
    block = np.squeeze(block)
    return block, whole_test_set_preproc, test_set[1], w_lr, b_lr


def eval_with_truthtables(data: list, w_lr: np.array, b_lr: object, block: np.array,
                          npatches: int, nfilters: int, dataset: str) -> None:
    """
    Evaluation on the test set with only the truth tables, no NN are loading
    @param dataset:
    @rtype: None
    @param data: test set
    @param w_lr: LR
    @param b_lr: bias
    @param block: truth table
    @param npatches: number of patches
    @param nfilters: number of filter
    """
    acc = 0
    data, labels = data[0], data[1]
    for i, inpt in enumerate(data):
        out_filt = []
        if dataset == "breast_cancer":
            for idx in range(inpt.shape[0]):
                out_filt.append([block[idx][inpt[idx]]].copy())
        else:
            for filt in range(nfilters):
                filt_in = inpt[filt * npatches:(filt + 1) * npatches]
                out_filt.append(block[filt, filt_in].copy())
        out_filt = np.concatenate(out_filt, axis=0)
        pred2 = np.argmax((w_lr @ out_filt).transpose() + b_lr)
        label = np.argmax(labels[i])
        acc += int(pred2 == label)
    print(f"\n Accuracy: {100 * acc / len(data)}% \n")


def load_config(config_name: str) -> (int, int, int, int, str, str, str, dict, int):
    """
    Load the configuration of the dataset
    @rtype: essential attribute for the run
    @param config_name: path of the config
    @return: essential attributes for the run
    """
    conf = Config(path=f"config/{config_name}")
    ksize = conf.model.kernel_size_per_block[0]
    nfilters = conf.model.Blocks_filters_output[0]
    pad = conf.model.padding_per_block[0]
    data_dir = conf.general.DATA_DIR
    dataset = config_name
    model_path = conf.eval.path_load_model
    mappings_path = conf.model.load_permut
    stride = conf.model.Blocks_strides[0]
    k_fold = conf.general.kfold
    return ksize, nfilters, pad, stride, data_dir, dataset, model_path, mappings_path, k_fold


def main() -> None:
    config_name = Config(path="config/").dataset
    ksize, nfilters, pad, stride, data_dir, dataset, model_path, mappings_path, k_fold = load_config(config_name)
    block, testset, labels, w_lr, b_lr = prepare_dataset(ksize, nfilters, pad, stride, data_dir, dataset, model_path,
                                                         mappings_path, k_fold)
    npatches = testset[0].shape[0] // nfilters
    eval_with_truthtables([testset, labels], w_lr, b_lr, block, npatches, nfilters, dataset)


if __name__ == '__main__':
    main()
