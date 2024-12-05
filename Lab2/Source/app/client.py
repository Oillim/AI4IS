L2_REG = 0.0001

import argparse
from config import SEND_RECEIVE_CONF as SRC
import socket
import time
import hmac
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from utils import *
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import data_processing as dp
import feature_extraction as fe

class FederatedClientCallback(Callback):
    def __init__(self, model, server_ip, client_index, x_val, y_val, data_size, fake=False):
        super().__init__()
        self._model = model
        self._client_index = client_index
        self._server_ip = server_ip.split(':')[0]
        self._server_port = int(server_ip.split(':')[1])
        self._data_size = data_size
        self.fake = fake
        self._get_task_index()  # Retrieve client and worker index from server
        self._receive_initial_weights()
        self.x_val = x_val
        self.y_val = y_val
    
    def _receive_initial_weights(self):
        """Receives initial model weights from server."""
        with self._connect_to_server() as worker_socket:
            initial_weights = self._get_np_array(worker_socket)
            self.model.set_weights(initial_weights)
            worker_socket.close()

    def _get_task_index(self):
        client_socket = self._start_socket_worker()
        self._send_np_array([self._client_index, self._data_size], client_socket)
        client_socket.recv(1024).decode('utf-8')
        client_socket.close()

    def _start_socket_worker(self):
        """Attempts connection to server with retries."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        max_retries = 20
        retry_delay = 2
        
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
            self._synchronize_weights(self.fake)

    def on_train_end(self, logs=None):
        with self._connect_to_server() as worker_socket:
            self._send_np_array([-1], worker_socket)
            worker_socket.close()

    def _synchronize_weights(self, fake=False, attack_strength=0.5):
        """
        Sends local model weights to server and receives aggregated weights.
        Optionally applies a model poisoning attack.
        
        Args:
            fake (bool): Whether to apply model poisoning (malicious behavior).
            attack_strength (float): Strength of the model poisoning attack.
        """
        print('\nSynchronizing weights...')
        with self._connect_to_server() as worker_socket:
            # Send client index
            self._send_np_array([self._client_index], worker_socket)
            
            if fake:
                poisoned_weights = []
                for weight in self.model.get_weights():
                    if np.issubdtype(weight.dtype, np.floating):  # Only perturb floating-point weights
                        noise = np.random.normal(0, attack_strength, weight.shape)  # Add Gaussian noise
                        poisoned_weights.append(weight + noise)
                    else:
                        poisoned_weights.append(weight)  # Non-floating weights are unchanged
                self._send_np_array(poisoned_weights, worker_socket)
            else:
                # Send unmodified weights
                self._send_np_array(self.model.get_weights(), worker_socket)
            
            # Receive aggregated weights from the server
            broadcasted_weights = self._get_np_array(worker_socket)
            self.model.set_weights(broadcasted_weights)
            worker_socket.close()

    def _synchronize_weights_base_acc(self, fake=False, attack_strength=0.5):
        """
        Sends local model weights to server and receives aggregated weights.
        Optionally applies a model poisoning attack.
        
        Args:
            fake (bool): Whether to apply model poisoning (malicious behavior).
            attack_strength (float): Strength of the model poisoning attack.
        """
        print('\nSynchronizing weights...')
        with self._connect_to_server() as worker_socket:
            # Send client index
            _, acc = self.model.evaluate(self.x_val, self.y_val, verbose=0)
            self._send_np_array([self._client_index], worker_socket)
            self._send_np_array([acc], worker_socket)
            
            if fake:
                poisoned_weights = []
                for weight in self.model.get_weights():
                    if np.issubdtype(weight.dtype, np.floating):  # Only perturb floating-point weights
                        noise = np.random.normal(0, attack_strength, weight.shape)  # Add Gaussian noise
                        poisoned_weights.append(weight + noise)
                    else:
                        poisoned_weights.append(weight)  # Non-floating weights are unchanged
                self._send_np_array(poisoned_weights, worker_socket)
            else:
                # Send unmodified weights
                self._send_np_array(self.model.get_weights(), worker_socket)
            
            # Receive aggregated weights from the server
            broadcasted_weights = self._get_np_array(worker_socket)
            self.model.set_weights(broadcasted_weights)
            worker_socket.close()
            
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

BATCH_SIZE = 32
N_EPOCHS = 6 
LR = 0.005
sampling = 3
def create_model(n_features, reg_method=None):
    """Creates a simple neural network model."""
    if reg_method == 'l2':
        model = Sequential([
            Input(shape=(n_features)),
            Flatten(),
            Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(L2_REG))
        ])
    else:
        model = Sequential([
            Input(shape=(n_features)),
            Flatten(),
            Dense(10, activation='softmax')
        ])
    model.compile(optimizer=Adam(learning_rate=LR), loss=SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    return model

def train_client(server_ip, client_index, attack=None, reg_method=None):
    """Loads data, preprocesses, creates model, and starts training with federated callback."""
    (x_train, y_train), (_, _) = dp.load_data_keras("../../Data")
    (x_train, y_train), (x_val, y_val) = dp.split_data(x_train, y_train, client_index)

    #(x_train, y_train), (x_val, y_val) = fe.HogPreprocess(x_train, y_train, x_val, y_val, test=False)
    (x_train, y_train), (x_val, y_val) = fe.ResnetPreprocess(x_train, y_train, x_val, y_val, sampling=sampling)
    if attack == 'data':
        (x_train, y_train) = fe.poison_dataset(x_train, y_train, 3, poison_ratio=0.5)
    
    n_features = x_train.shape[1:]
    model = create_model(n_features, reg_method)

    if attack == 'model':
        client_callback = FederatedClientCallback(model, server_ip, client_index, x_val, y_val, x_train.shape[0], fake=True)
        print(f"Applying model poisoning attack to client {client_index}")
    else:
        client_callback = FederatedClientCallback(model, server_ip, client_index, x_val, y_val, x_train.shape[0])
    
    checkpoint = ModelCheckpoint(
        f'../model/client_{client_index}.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode = 'max',
        verbose=0
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=N_EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=[client_callback, checkpoint]
    )

    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Client Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'client_{client_index}_loss.png')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Federated learning client.")
    parser.add_argument("--client_index", type=int, required=True, help="Server IP and port in the format 'IP:PORT'")
    parser.add_argument("--attack", type=str, required=False, help="Whether to attack a model at client ('data' for data poisoning, 'model' for model poisoning)")
    parser.add_argument("--reg_method", type=str, required=False, help="Regularization method")
    args = parser.parse_args()
    server_ip = '127.0.0.1:5000'

    train_client(server_ip, args.client_index, args.attack, args.reg_method)
