import os 
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
from tensorflow.keras.optimizers import Adam
from config import SEND_RECEIVE_CONF as SRC
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import data_processing as dp
import feature_extraction as fe

LR = 0.005
sampling = 3
highest_acc = 0.0
class FederatedServer:
    """Server for Federated Learning without local training."""

    def __init__(self, model, private_ip, wait_time=300):
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
        self._client_infor = {}

        while (time.time() < t_end) and (len(users) < 3):
            try:
                sock, _ = self._server_socket.accept()
                [client_index, data_size] = self._get_np_array(sock)
                self._client_infor[client_index] = data_size
                print('Worker connected: ', client_index)
                users.append(sock) 
            except socket.timeout:
                pass
        print(self._client_infor)
        
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
        gathered_weights = {}
        users = []

        # Kết nối với các worker
        for _ in range(self.num_workers):
            try:
                sock, _ = self._server_socket.accept()
                [client_index] = self._get_np_array(sock)
                # Kiểm tra xem worker có ngắt kết nối không
                if (client_index == -1):
                    self.num_workers -= 1
                    print('Worker disconnected. Total workers: ', self.num_workers)
                    continue
                else:
                    print('Worker connected: ', client_index)
                received = self._get_np_array(sock)
                gathered_weights[client_index] = received
                users.append(sock)
            except (socket.timeout, ConnectionResetError, BrokenPipeError):
                print("Connection error or timeout.")
                break

        total_data_size = sum(self._client_infor[client_index] for client_index in gathered_weights.keys())
        
        if total_data_size == 0:  # Tránh chia cho 0
            return self._model.get_weights()

        first_weights = []
        second_weights = []

        for client_index, weights in gathered_weights.items():
            first_weights.append(weights[0])  # Mảng 2 chiều
            second_weights.append(weights[1])  # Mảng 1 chiều

        averaged_weights_2d = np.zeros_like(first_weights[0])  
        averaged_weights_1d = np.zeros_like(second_weights[0])  

        for client_index in gathered_weights.keys():
            weight = self._client_infor[client_index] / total_data_size  
            averaged_weights_2d += first_weights.pop(0) * weight
            averaged_weights_1d += second_weights.pop(0) * weight

        for user in users:
            self._send_np_array([averaged_weights_2d, averaged_weights_1d], user)  # Gửi cả hai mảng
            user.close()

        self._model.set_weights([averaged_weights_2d, averaged_weights_1d])
        loss, accuracy = self._model.evaluate(x_val, y_val, verbose=0)
        
        print(f"Validation on CIFAR-10 - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        global highest_acc
        if accuracy > highest_acc:
            highest_acc = accuracy
            self._model.save('../model/federate_learning_model.keras')
        
        print("Model weights updated after aggregation.")
    
        if self.num_workers == 0:
            print('No workers connected. Exiting...')
            return

    def close_server(self):
        """Close the server socket."""
        self._server_socket.close()



def create_model(n_features):
    model = Sequential([
        Input(shape=(n_features)),
        Flatten(),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=LR), loss=SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    return model



def train_server(server_ip):
    (x_train, y_train), (x_test, y_test) = dp.load_data_keras("../../Data")
    # (x_train, y_train), (x_test, y_test) = fe.HogPreprocess(x_train, y_train, x_test, y_test, test=False)

    (x_test, y_test) = fe.ResnetPreprocess(x_test=x_test, y_test=y_test, sampling=sampling, test=True)
    x_val = x_test[x_test.shape[0] // 2:]
    y_val = y_test[y_test.shape[0] // 2:]

    n_features = x_test.shape[1:]
    model = create_model(n_features)

    server = FederatedServer(model, server_ip)
    if (server.num_workers == 0):
        print('No workers connected. Exiting...')
        return
    
    server.distribute_initial_model()

    for round in range(100):  
        print(f"\n--- Round {round + 1} ---")
        server.aggregate_updates(x_val, y_val)
        if server.num_workers == 0:
            break
    server.close_server()


if __name__ == '__main__':
    server_ip = '127.0.0.1:5000'
    train_server(server_ip)
