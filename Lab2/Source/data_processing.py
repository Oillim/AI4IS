import pickle
import numpy as np
import torch
import os
from keras.src import backend
#unpickle the data from the file
def load_batch(fpath, label_key="labels"):
    """Internal utility for parsing CIFAR data.

    Args:
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.

    Returns:
        A tuple `(data, labels)`.
    """
    with open(fpath, "rb") as f:
        d = pickle.load(f, encoding="bytes")
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode("utf8")] = v
        d = d_decoded
    data = d["data"]
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels

def load_data_keras(path):
    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype="uint8")
    y_train = np.empty((num_train_samples,), dtype="uint8")
    
    # batches are within an inner folder
    path = os.path.join(path, "cifar-10-batches-py")
    for i in range(1, 6):
        fpath = os.path.join(path, "data_batch_" + str(i))
        (
            x_train[(i - 1) * 10000 : i * 10000, :, :, :],
            y_train[(i - 1) * 10000 : i * 10000],
        ) = load_batch(fpath)

    fpath = os.path.join(path, "test_batch")
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if backend.image_data_format() == "channels_last":
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    x_test = x_test.astype(x_train.dtype)
    y_test = y_test.astype(y_train.dtype)

    return (x_train, y_train), (x_test, y_test)        

def split_data(x, y):
    client_classes = {
        "client1": [0, 1, 2, 3, 4],
        "client2": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "client3": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    }

    client_data = {
        "client1": {"x": [], "y": []},
        "client2": {"x": [], "y": []},
        "client3": {"x": [], "y": []}
    }

    for class_id in np.unique(y):

        class_indices = np.where(y == class_id)[0]
        clients_with_class = [client for client, classes in client_classes.items() if class_id in classes]
        num_clients = len(clients_with_class)
        np.random.shuffle(class_indices)
        split_indices = np.array_split(class_indices, num_clients)

        for client, indices in zip(clients_with_class, split_indices):
            client_data[client]["x"].append(x[indices])
            client_data[client]["y"].append(y[indices])

    for client in client_data:
        client_data[client]["x"] = np.concatenate(client_data[client]["x"], axis=0)
        client_data[client]["y"] = np.concatenate(client_data[client]["y"], axis=0)

    for client in client_data:
        print(f"{client} - x shape: {client_data[client]['x'].shape}, y shape: {client_data[client]['y'].shape}")
    
    return client_data

if __name__ == "__main__":
    print("Load CIFAR10 from local...")
