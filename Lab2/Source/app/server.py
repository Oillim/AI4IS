import os 
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import socket
import ssl
import hmac
import numpy as np
import pickle
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from skimage.feature import hog
from config import SEND_RECEIVE_CONF as SRC
from skimage.color import rgb2gray


class FederatedServer:
    """Server for Federated Learning without local training."""

    def __init__(self, model, private_ip, wait_time=15):
        self._model = model
        self._private_ip = private_ip.split(':')[0]
        self._private_port = int(private_ip.split(':')[1])
        self._wait_time = wait_time
        self.num_workers = self._get_task_index()

    
    def _start_socket_server(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self._private_ip, self._private_port))
        server_socket.listen()
        return server_socket

    def _get_task_index(self):
        print('Waiting for workers to connect...')
        self._server_socket = self._start_socket_server()
        self._server_socket.settimeout(5)
        users = []
        t_end = time.time() + self._wait_time

        while time.time() < t_end:
            try:
                sock, _ = self._server_socket.accept()
                client_index = int(sock.recv(1024).decode('utf-8'))
                print('Worker connected: ', client_index)
                users.append(sock)
            except socket.timeout:
                pass

        num_workers = len(users)
        _ = [us.send("Server accepted".encode('utf-8')) \
        for i, us in enumerate(users)]
        self._nex_task_index = len(users) + 1
        _ = [us.close() for us in users]

        self._server_socket.settimeout(5)
        return num_workers

    @staticmethod
    def _receiving_subroutine(connection_socket):
        """Receive numpy arrays securely from a client."""
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
        """Receive and decode numpy array."""
        message = self._receiving_subroutine(connection_socket)
        final_image = pickle.loads(message)
        return final_image

    @staticmethod
    def _send_np_array(arrays_to_send, connection_socket):
        """Send a list of numpy arrays securely."""
        serialized = pickle.dumps(arrays_to_send)
        signature = hmac.new(SRC.key, serialized, SRC.hashfunction).digest()
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

    def distribute_initial_model(self):
        initial_weights = self._model.get_weights()
        users = []

        for _ in range(self.num_workers):
            try:
                sock, _ = self._server_socket.accept()
                users.append(sock)
            except socket.timeout:
                break

        for user in users:
            self._send_np_array(initial_weights, user)
            user.close()
        print("Initial model distributed to clients.")

    def aggregate_updates(self, x_val, y_val):
        gathered_weights = [self._model.get_weights()]
        users = []

        for _ in range(self.num_workers):
            try:
                sock, _ = self._server_socket.accept()
                received = self._get_np_array(sock)
                if (received == [-1]):
                    self.num_workers -= 1
                    print('Worker disconnected. Total workers: ', self.num_workers)
                    continue

                gathered_weights.append(received)
                users.append(sock)
            except (socket.timeout, ConnectionResetError, BrokenPipeError):
                break

        averaged_weights = [np.mean([layer[i] for layer in gathered_weights], axis=0)
                            for i in range(len(gathered_weights[0]))]
        
        for user in users:
            self._send_np_array(averaged_weights, user)
            user.close()
        
        self._model.set_weights(averaged_weights)
        loss, accuracy = self._model.evaluate(x_val, y_val, verbose=0)
        print(f"Validation on CIFAR-10 - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        print("Model weights updated after aggregation.")

    def close_server(self):
        """Close the server socket."""
        self._server_socket.close()


def create_model():
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

def preprocess_data(x_train, y_train, x_test, y_test):
    """Processes data to prepare for training."""
    _x_train = np.array([feature_extraction(x_train[i]) for i in range(len(x_train))])
    _y_train = np.array([y_train[i] for i in range(len(y_train))])

    _x_test = np.array([feature_extraction(x_test[i]) for i in range(len(x_test))])
    _y_test = np.array([y_test[i] for i in range(len(y_test))])

    return (_x_train, _y_train), (_x_test, _y_test)


def train_server(server_ip):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = [ rgb2gray(x_train[i]) for i in range(len(x_train))]
    x_test = [ rgb2gray(x_test[i]) for i in range(len(x_test))]
    (x_train, y_train), (x_test, y_test) = preprocess_data(x_train, y_train, x_test, y_test)
    x_val = x_test[x_test.shape[0] // 2:]
    y_val = y_test[y_test.shape[0] // 2:]

    model = create_model()
    
    server = FederatedServer(model, server_ip)
    if (server.num_workers == 0):
        print('No workers connected. Exiting...')
        return
    
    server.distribute_initial_model()

    for round in range(100):  
        print(f"\n--- Round {round + 1} ---")
        server.aggregate_updates(x_val, y_val)

    server.close_server()


if __name__ == '__main__':
    server_ip = '127.0.0.1:5000'
    train_server(server_ip)
