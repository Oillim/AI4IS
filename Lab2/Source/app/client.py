

import argparse
from config import SEND_RECEIVE_CONF as SRC
import socket
import time
import hmac
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from skimage.feature import hog
from skimage.color import rgb2gray


class FederatedClientCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, server_ip, client_index):
        super().__init__()
        self._model = model
        self._client_index = client_index
        self._server_ip = server_ip.split(':')[0]
        self._server_port = int(server_ip.split(':')[1])
        self._get_task_index()  # Retrieve client and worker index from server
        self._receive_initial_weights()
    
    def _receive_initial_weights(self):
        """Receives initial model weights from server."""
        with self._connect_to_server() as worker_socket:
            initial_weights = self._get_np_array(worker_socket)
            self.model.set_weights(initial_weights)
            worker_socket.close()

    def _get_task_index(self):
        client_socket = self._start_socket_worker()
        client_socket.send(str(self._client_index).encode('utf-8'))
        client_socket.recv(1024).decode('utf-8')
        client_socket.close()

    def _start_socket_worker(self):
        """Attempts connection to server with retries."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        max_retries = 5
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                print(f'Attempting to connect to server (attempt {attempt + 1})...')
                sock.connect((self._server_ip, self._server_port))
                print('Connected to server.')
                return sock
            except (socket.error, ConnectionRefusedError):
                print(f'Connection failed. Retrying in {retry_delay} seconds...')
                time.sleep(retry_delay)
        
        raise ConnectionError("Could not connect to the server after multiple attempts.")

    @property
    def model(self):
        """Returns the Keras model."""
        return self._model

    def _connect_to_server(self):
        """Creates socket connection to the server."""
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((self._server_ip, self._server_port))
        return client_socket

    @staticmethod
    def _receiving_subroutine(connection_socket):
        """Handles receiving serialized data from the server with HMAC verification."""
        timeout = 0.5
        while True:
            ultimate_buffer = b''
            connection_socket.settimeout(240)
            first_round = True
            while True:
                try:
                    receiving_buffer = connection_socket.recv(SRC.buffer)
                except socket.timeout:
                    break
                if first_round:
                    connection_socket.settimeout(timeout)
                    first_round = False
                if not receiving_buffer:
                    break
                ultimate_buffer += receiving_buffer

            pos_signature = SRC.hashsize
            signature = ultimate_buffer[:pos_signature]
            message = ultimate_buffer[pos_signature:]
            good_signature = hmac.new(SRC.key, message, SRC.hashfunction).digest()

            if signature != good_signature:
                connection_socket.send(SRC.error)
                timeout += 0.5
                continue
            else:
                connection_socket.send(SRC.recv)
                connection_socket.settimeout(120)
                return message

    def _get_np_array(self, connection_socket):
        """Receives list of numpy arrays from server."""
        message = self._receiving_subroutine(connection_socket)
        final_image = pickle.loads(message)
        return final_image

    @staticmethod
    def _send_np_array(arrays_to_send, connection_socket):
        """Sends list of numpy arrays with HMAC signature."""
        serialized = pickle.dumps(arrays_to_send)
        signature = hmac.new(SRC.key, serialized, SRC.hashfunction).digest()
        assert len(signature) == SRC.hashsize
        message = signature + serialized
        connection_socket.settimeout(240)
        connection_socket.sendall(message)
        while True:
            check = connection_socket.recv(len(SRC.error))
            if check == SRC.error:
                connection_socket.sendall(message)
            elif check == SRC.recv:
                connection_socket.settimeout(120)
                break

    def on_train_batch_end(self, batch, logs=None):
        """Sync weights with server every 100 batches."""
        if batch % 100 == 0 and batch > 0:
            self._synchronize_weights()

    def on_train_end(self, logs=None):
        with self._connect_to_server() as worker_socket:
            self._send_np_array([-1], worker_socket)
            worker_socket.close()

    def _synchronize_weights(self):
        """Sends local model weights to server and receives aggregated weights."""
        print('\nSynchronizing weights...')
        with self._connect_to_server() as worker_socket:
            self._send_np_array(self.model.get_weights(), worker_socket)
            broadcasted_weights = self._get_np_array(worker_socket)
            self.model.set_weights(broadcasted_weights)
            worker_socket.close()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.losses import SparseCategoricalCrossentropy

def create_model():
    """Creates a simple neural network model."""
    model = Sequential([
        Input(shape=(324,)),
        Flatten(),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    return model

normalize = True          
block_norm = 'L2-Hys'     
orientations = 9          
pixels_per_cell = [8, 8]  
cells_per_block = [2, 2]  

def feature_extraction(x):
    """Extracts features using Histogram of Oriented Gradients (HOG)."""
    return hog(x, orientations, pixels_per_cell, cells_per_block, block_norm, visualize=False, transform_sqrt=normalize)

def preprocess_data(x_train, y_train, x_val, y_val):
    """Processes data to prepare for training."""
    _x_train = np.array([feature_extraction(x_train[i]) for i in range(len(x_train))])
    _y_train = np.array([y_train[i] for i in range(len(y_train))])

    _x_val = np.array([feature_extraction(x_val[i]) for i in range(len(x_val))])
    _y_val = np.array([y_val[i] for i in range(len(y_val))])

    return (_x_train, _y_train), (_x_val, _y_val)

import numpy as np

def split_data(x, y, client_index):
    client_classes = {
        0: [0, 1, 2, 3, 4],   # Classes for client 0
        1: [1, 2, 3, 4, 5, 6, 7, 8, 9],  # Classes for client 1
        2: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Classes for client 2
    }

    client_data = {
        0: {"x": [], "y": []},  # Data for client 0
        1: {"x": [], "y": []},  # Data for client 1
        2: {"x": [], "y": []}   # Data for client 2
    }

    if client_index not in client_data:
        raise ValueError("Invalid client index")
    
    for class_id in np.unique(y):
        class_indices = np.where(y == class_id)[0]
        
        clients_with_class = [client for client, classes in client_classes.items() if class_id in classes]
        num_clients = len(clients_with_class)

        np.random.shuffle(class_indices)
        split_indices = np.array_split(class_indices, num_clients)

        for client, indices in zip(clients_with_class, split_indices):
            client_data[client]["x"].append(x[indices])
            client_data[client]["y"].append(y[indices])

    x_client = np.concatenate(client_data[client_index]["x"], axis=0)
    y_client = np.concatenate(client_data[client_index]["y"], axis=0)

    # shuffle
    indices = np.random.permutation(len(x_client))
    x_client = x_client[indices]
    y_client = y_client[indices]

    train_size = int(0.9 * len(x_client))
    x_train, x_val = x_client[:train_size], x_client[train_size:]
    y_train, y_val = y_client[:train_size], y_client[train_size:]

    print(f"Client {client_index} - x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"Client {client_index} - x_val shape: {x_val.shape}, y_val shape: {y_val.shape}")

    return (x_train, y_train), (x_val, y_val)



def train_client(server_ip, client_index):
    """Loads data, preprocesses, creates model, and starts training with federated callback."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    (x_train, y_train), (x_val, y_val) = split_data(x_train, y_train, client_index)

    x_train = [ rgb2gray(x_train[i]) for i in range(len(x_train))]
    x_val = [ rgb2gray(x_val[i]) for i in range(len(x_val))]

    (x_train, y_train), (x_val, y_val) = preprocess_data(x_train, y_train, x_val, y_val)
    model = create_model()
    client_callback = FederatedClientCallback(model, server_ip, client_index)

    model.fit(
        x_train,
        y_train,
        batch_size=32,
        epochs=3,
        validation_data=(x_val, y_val),
        callbacks=[client_callback]
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Federated learning client.")
    parser.add_argument("--client_index", type=int, required=True, help="Server IP and port in the format 'IP:PORT'")
    args = parser.parse_args()

    server_ip = '127.0.0.1:5000'

    train_client(server_ip, args.client_index)
