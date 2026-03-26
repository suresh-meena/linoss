"""
This script processes the UEA datasets and saves the processed data in the data_dir/processed directory.
It has been adapted from https://github.com/jambo6/neuralRDEs
"""

import argparse
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sktime.datasets import load_from_arff_to_dataframe
from tqdm import tqdm


def save_pickle(obj, filename):
    """Saves a pickle object."""
    with open(filename, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    """Loads a pickle object."""
    with open(filename, "rb") as handle:
        return pickle.load(handle)


def _to_numpy_pickle_value(value):
    """Recursively converts legacy pickle payloads to NumPy-backed values."""
    if isinstance(value, tuple):
        return tuple(_to_numpy_pickle_value(item) for item in value)
    if isinstance(value, list):
        return [_to_numpy_pickle_value(item) for item in value]
    return np.ascontiguousarray(np.asarray(value))


def create_numpy_data(train_file, test_file):
    """Creates NumPy tensors for test and training from the UCR arff format.

    Args:
        train_file (str): The location of the training data arff file.
        test_file (str): The location of the testing data arff file.

    Returns:
        data_train, data_test, labels_train, labels_test: All as NumPy arrays.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        train_data, train_labels = load_from_arff_to_dataframe(train_file)
        test_data, test_labels = load_from_arff_to_dataframe(test_file)

    def convert_data(data):
        data_expand = data.map(lambda x: x.values).values
        return np.ascontiguousarray(
            np.stack([np.vstack(x).T for x in data_expand]),
            dtype=np.float32,
        )

    train_data, test_data = convert_data(train_data), convert_data(test_data)

    encoder = LabelEncoder().fit(train_labels)
    train_labels = np.asarray(encoder.transform(train_labels), dtype=np.int64)
    test_labels = np.asarray(
        encoder.transform(test_labels),
        dtype=np.int64,
    )

    return train_data, test_data, train_labels, test_labels


def convert_all_files(data_dir, *, overwrite=False, datasets=None):
    """Convert UEA files into NumPy-backed processed data."""
    arff_folder = os.path.join(data_dir, "raw", "UEA", "Multivariate_arff")
    dataset_filter = set(datasets) if datasets else None

    for ds_name in tqdm(
        sorted(
            x
            for x in os.listdir(arff_folder)
            if os.path.isdir(os.path.join(arff_folder, x))
        )
    ):
        if dataset_filter is not None and ds_name not in dataset_filter:
            continue

        train_file = os.path.join(arff_folder, ds_name, f"{ds_name}_TRAIN.arff")
        test_file = os.path.join(arff_folder, ds_name, f"{ds_name}_TEST.arff")

        save_dir = os.path.join(data_dir, "processed", "UEA", ds_name)

        if any(
            [
                os.path.basename(x)
                not in os.listdir(os.path.join(arff_folder, ds_name))
                for x in (train_file, test_file)
            ]
        ):
            if ds_name not in ["Images", "Descriptions"]:
                print("No files found for folder: {}".format(ds_name))
            continue
        elif os.path.isdir(save_dir) and not overwrite:
            print("Files already exist for: {}".format(ds_name))
            continue
        else:
            os.makedirs(save_dir, exist_ok=True)
            train_data, test_data, train_labels, test_labels = create_numpy_data(
                train_file, test_file
            )
            data = np.concatenate([train_data, test_data], axis=0)
            labels = np.concatenate([train_labels, test_labels], axis=0)

            _, indices, inverse_indices = np.unique(
                data, axis=0, return_index=True, return_inverse=True
            )
            data = np.ascontiguousarray(data[indices], dtype=np.float32)
            labels = np.ascontiguousarray(labels[indices], dtype=np.int64)
            print(
                f"Deleting {len(inverse_indices) - len(indices)} repeated samples in {ds_name}"
            )

            original_idxs = (
                np.arange(0, train_data.shape[0], dtype=np.int64),
                np.arange(train_data.shape[0], data.shape[0], dtype=np.int64),
            )

            save_pickle(data, os.path.join(save_dir, "data.pkl"))
            save_pickle(labels, os.path.join(save_dir, "labels.pkl"))
            save_pickle(original_idxs, os.path.join(save_dir, "original_idxs.pkl"))


def rewrite_processed_pickles_to_numpy(data_dir, *, datasets=None):
    """Rewrites existing processed UEA pickles in place as NumPy-backed payloads."""
    processed_root = os.path.join(data_dir, "processed", "UEA")
    dataset_filter = set(datasets) if datasets else None
    rewrite_candidates = (
        "data.pkl",
        "labels.pkl",
        "original_idxs.pkl",
        "X_train.pkl",
        "y_train.pkl",
        "X_val.pkl",
        "y_val.pkl",
        "X_test.pkl",
        "y_test.pkl",
    )

    for ds_name in sorted(
        x
        for x in os.listdir(processed_root)
        if os.path.isdir(os.path.join(processed_root, x))
    ):
        if dataset_filter is not None and ds_name not in dataset_filter:
            continue
        save_dir = os.path.join(processed_root, ds_name)
        rewritten = []
        for filename in rewrite_candidates:
            path = os.path.join(save_dir, filename)
            if not os.path.exists(path):
                continue
            save_pickle(_to_numpy_pickle_value(load_pickle(path)), path)
            rewritten.append(filename)
        if rewritten:
            print(f"Rewrote NumPy pickles for {ds_name}: {', '.join(rewritten)}")


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Process or rewrite UEA datasets into NumPy-backed pickle files."
    )
    parser.add_argument(
        "--data-dir",
        default="data_dir",
        help="Base data directory containing raw/ and processed/ trees.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate processed UEA datasets even when processed directories already exist.",
    )
    parser.add_argument(
        "--rewrite-existing",
        action="store_true",
        help="Rewrite existing processed UEA pickles in place as NumPy-backed payloads.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Limit processing or rewriting to one or more dataset names.",
    )
    return parser


def main():
    args = _build_arg_parser().parse_args()
    if args.overwrite or not args.rewrite_existing:
        convert_all_files(
            args.data_dir,
            overwrite=args.overwrite,
            datasets=args.datasets,
        )
    if args.rewrite_existing:
        rewrite_processed_pickles_to_numpy(
            args.data_dir,
            datasets=args.datasets,
        )


if __name__ == "__main__":
    main()
