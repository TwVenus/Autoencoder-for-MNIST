import random
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from sklearn.metrics import mean_squared_error, accuracy_score

class Readfile(object):
    def __init__(self):
        (X_train_image, y_train_label), (X_test_image, y_test_label) = mnist.load_data()
        self.feature_list = np.array([X_train_image[i].reshape(-1) for i in range(0, X_train_image.shape[0])])

class Write_np(object):
    @staticmethod
    def save(filename, data):
        np.save(filename, data)

class BPNN(object):
    def __init__(self, dataset, hidden_later1_node, hidden_later2_node, hidden_later3_node, learning_rate, batch_size, epoch, momentum):
        self.feature_list = dataset.feature_list
        self.input_node = dataset.feature_list.shape[1]
        self.hidden_later1_node = hidden_later1_node
        self.hidden_later2_node = hidden_later2_node
        self.hidden_later3_node = hidden_later3_node
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch = epoch
        self.momentum = momentum

    # step1: definite of network architecture
    def set_nn_architecture(self):
        # weight
        self.weight_list_h = [np.random.uniform(-1.0, 1.0, size=(self.input_node, self.hidden_later1_node)),
                              np.random.uniform(-1.0, 1.0, size=(self.hidden_later1_node, self.hidden_later2_node)),
                              np.random.uniform(-1.0, 1.0, size=(self.hidden_later2_node, self.hidden_later3_node))]
        self.weight_list_o = [np.random.uniform(-1.0, 1.0, size=(self.hidden_later1_node, self.input_node)),
                              np.random.uniform(-1.0, 1.0, size=(self.hidden_later2_node, self.hidden_later1_node)),
                              np.random.uniform(-1.0, 1.0, size=(self.hidden_later3_node, self.hidden_later2_node))]

        # bias = 1 ,
        self.bias_weight_h = [np.random.uniform(-1.0, 1.0, size=self.hidden_later1_node),
                              np.random.uniform(-1.0, 1.0, size=self.hidden_later2_node),
                              np.random.uniform(-1.0, 1.0, size=self.hidden_later3_node)]
        self.bias_weight_o = [np.random.uniform(-1.0, 1.0, size=self.input_node),
                              np.random.uniform(-1.0, 1.0, size=self.hidden_later1_node),
                              np.random.uniform(-1.0, 1.0, size=self.hidden_later2_node)]

        # momentum parameter
        self.pre_delta_o = 0
        self.pre_delta_h = 0
        self.pre_delta_o_bias = 0
        self.pre_delta_h_bias = 0

    def sigmoid(self, value):
        return 1 / (1 + np.exp(-1*value))

    def Relu(self, value):
        return value * (value > 0)

    def forward(self, mini_x, layer_num):
        self.after_LF_h = self.sigmoid((np.dot(mini_x, self.weight_list_h[layer_num]) + self.bias_weight_h[layer_num]))
        self.after_LF_o = self.sigmoid((np.dot(self.after_LF_h, self.weight_list_o[layer_num]) + self.bias_weight_o[layer_num]))

    # step2: feed_forward stage (learning function: sigmoid)
    def predict(self, mini_x, mini_y, layer_num):
        self.forward(mini_x, layer_num)

        self.mse = 0
        self.mse = mean_squared_error(self.after_LF_o, mini_y)

    # step3: back propagation neural network
    def backend(self, mini_x, mini_y, layer_num):
        E = (mini_y - self.after_LF_o)
        delta_y = E * self.after_LF_o * (1 - self.after_LF_o) #500 200
        delta_h = self.after_LF_h * (1 - self.after_LF_h) * np.dot(delta_y, self.weight_list_o[layer_num].T) #500  200

        self.weight_list_o[layer_num] += self.learning_rate * self.after_LF_h.T.dot(delta_y) + self.momentum * self.pre_delta_o
        self.weight_list_h[layer_num] += self.learning_rate * mini_x.T.dot(delta_h) + self.momentum * self.pre_delta_h
        self.bias_weight_o[layer_num] += self.learning_rate * delta_y.sum() + self.momentum * self.pre_delta_o_bias
        self.bias_weight_h[layer_num] += self.learning_rate * delta_h.sum() + self.momentum * self.pre_delta_h_bias

        self.pre_delta_o = self.learning_rate * self.after_LF_h.T.dot(delta_y)
        self.pre_delta_h = self.learning_rate * mini_x.T.dot(delta_h)
        self.pre_delta_o_bias = self.learning_rate * delta_y.sum()
        self.pre_delta_h_bias = self.learning_rate * delta_h.sum()

    def train(self):
        self.after_sigmoid_list = []
        mse_threshold = [0.01, 0.0000001, 0.0000001]
        for hidden_layer_num in range(0, 3):
            self.set_nn_architecture()
            if hidden_layer_num == 0:
                self.input_list = np.where(self.feature_list > 0, 1, 0)
            else:
                self.input_list = np.where(np.array(self.after_sigmoid_list[hidden_layer_num - 1]) > 0, 1, 0)

            for _iter in range(0, self.epoch):
                for i in range(0, self.input_list.shape[0], self.batch_size):
                    mini_x = self.input_list[i: i+self.batch_size]
                    mini_y = self.input_list[i: i+self.batch_size]
                    self.predict(mini_x, mini_y, hidden_layer_num)
                    self.backend(mini_x, mini_y, hidden_layer_num)
                self.predict(self.input_list, self.input_list, hidden_layer_num)

                if(_iter % 5 == 0):
                    print("Epoch = {} ,train mse = {}".format(_iter, self.mse))

                if (self.mse <= mse_threshold[hidden_layer_num]):
                    self.after_sigmoid_list.append(self.after_LF_h)
                    Write_np.save("hidden" + str(hidden_layer_num) + "_weight", self.weight_list_h[hidden_layer_num])
                    Write_np.save("bias" + str(hidden_layer_num) + "_weight", self.bias_weight_h[hidden_layer_num])
                    Write_np.save("hidden_layer" + str(hidden_layer_num), self.after_LF_h)
                    print("Epoch_ = {} , train mse = {}".format(_iter, self.mse))
                    break

if __name__ == "__main__":
    dataset = Readfile()
    bpnn = BPNN(dataset, hidden_later1_node=200, hidden_later2_node=100, hidden_later3_node=50, learning_rate=0.002, batch_size=300, epoch=5000, momentum=0.99)
    bpnn.train()
