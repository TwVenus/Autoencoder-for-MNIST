# Autoencoder-for-MNIST
---
　
#### 一、概述
```
本程式為使用python語法。使用Auto Encoder決定Deep neural network的weight數值，當各層MSE小於0.01收斂；最後做BPNN，直到正確大於98即收斂。(學習函數為：Sigmoid，調整權重方法為：BPNN，使用mini batch)
```

#### 二、資料集：
```
手寫辨識(MNIST)
```

#### 三、網路架構 (Input node：784，Output node：10，Hidden1 node：200，Hidden2 node：100，Hidden3 node：50)

![network](https://i.imgur.com/JUQjw6X.jpg)

#### 四、Hyper parameter

![HP_table](https://i.imgur.com/iiCIUlX.jpg)

#### 五、程式概述 
###### 1. Read file
　
```
class Readfile(object):
    def __init__(self):
        (X_train_image, y_train_label), (X_test_image, y_test_label) = mnist.load_data()
        self.feature_list = np.array([X_train_image[i].reshape(-1) for i in range(0, X_train_image.shape[0])])
```

###### 2. Definite of network architecture
　
```
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
```

###### 3. Feed_forward stage (learning function: sigmoid)
　
```
 def predict(self, mini_x, mini_y, layer_num):
        self.forward(mini_x, layer_num)

        self.mse = 0
        self.mse = mean_squared_error(self.after_LF_o, mini_y)
```

###### 4. Back propagation neural network
　
```
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
```

###### 5. Train step
　
```
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
```

#### 六、結果
*Train accuracy = 99.00， Test accuracy = 95.90， Time = 707.8513(s)*

![result](https://i.imgur.com/wQKVVxF.jpg)








