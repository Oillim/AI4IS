import pickle
import numpy as np
import torch

#unpickle the data from the file
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#load data from the data_dir
def load_data(data_dir):
    
    cifar_train_data = None
    cifar_train_labels = []

    for i in range(1,6):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))

        cifar_train_labels += cifar_train_data_dict[b'labels']
    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))

    cifar_train_labels = np.array(cifar_train_labels)

    #convert the data to tensor
    tensor_data = torch.tensor(cifar_train_data, dtype=torch.float32)
    tensor_data /= 255.0

    #convert the labels to tensor
    tensor_labels = torch.tensor(cifar_train_labels, dtype=torch.int8)
    
    return tensor_data, tensor_labels

        


if __name__ == "__main__":
    print("Data Processing")