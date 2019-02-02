import numpy as np
import pickle
import pandas as pd
import copy

# import matplotlib.pyplot as plt

config = {}
config['layer_specs'] = [784, 50,
                         10]  # The length of list denotes number of hidden layers; each element denotes number of neurons in that layer; first element is the size of input layer, last element is the size of output layer.
config[
    'activation'] = 'sigmoid'  # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
config['batch_size'] = 1000  # Number of training samples per batch to be passed to network
config['epochs'] = 500  # Number of epochs to train the model
config['early_stop'] = True  # Implement early stopping or not
config['early_stop_epoch'] = 5  # Number of epochs for which validation loss increases to be counted as overfitting
config['L2_penalty'] = 0  # Regularization constant
config['momentum'] = True  # Denotes if momentum is to be applied or not
config['momentum_gamma'] = 0.9  # Denotes the constant 'gamma' in momentum expression
config['learning_rate'] = 0.0001  # Learning rate of gradient descent algorithm


# check vstack or hstack

def add_bias_column(x):
    ones = np.ones((1, len(x)))
    x = np.hstack((ones.T, x))
    return x


# def softmax(x):
#     """
#   Write the code for softmax activation function that takes in a numpy array and returns a numpy array.
#   """
#     output = np.exp(x) / np.sum(np.exp(x), axis=0)
#     return output


def softmax(x):
    """
    Write the code for softmax activation function that takes in a numpy array and returns a numpy array.
    """
    # in order to have better numerical stability, it is good to subtract
    # x with its maximum value
    x = x - np.max(x)
    # Let's now compute the exponentials of each entry in the vector
    e_x = np.exp(x)
    # Denominator is nothing but the sum of exponentials
    if e_x.ndim == 1:
        ex_sum = e_x.sum()
    else:
        # we will compute sums for each row separately
        ex_sum = np.sum(e_x, axis=1, keepdims=True)
    # finally the output of softmax function
    return e_x / ex_sum


def one_hot_encoder(input_labels_digits):
    classes_count = config['layer_specs'][-1]  # # [0,1,2,3,4,5] possible for 6 classes
    input_labels_digits = np.array(input_labels_digits).reshape(-1)  # [[0,0,1,1,2,2,3,3...]]
    one_hot_targets = np.eye(classes_count)[input_labels_digits.astype(dtype=int)]
    return one_hot_targets


# def plotting_func1(x, y, xlabel="epochs", ylabel="error", y1legend="validation accuracy",
#                    title="Accuracy vs epochs curve", ):
#     """
#     :param title:
#     :param x:
#     :param y:
#     :param y2:
#     :param xlabel:
#     :param ylabel:
#     :param y1legend:
#     :param y2legend:
#     :return:
#     """
#     plt.figure()
#     plt.plot(x, y)
#     #     plt.plot(x, y2)
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#
#     plt.grid(True)
#     plt.legend((y1legend), loc='best')
#     plt.show()
#
#
# def plotting_func2(x, y, y2, xlabel="epochs", ylabel="error", y1legend="validation accuracy", y2legend="training error",
#                    title="Accuracy vs epochs curve", ):
#     """
#     :param title:
#     :param x:
#     :param y:
#     :param y2:
#     :param xlabel:
#     :param ylabel:
#     :param y1legend:
#     :param y2legend:
#     :return:
#     """
#     plt.figure()
#     plt.plot(x, y)
#     plt.plot(x, y2)
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#
#     plt.grid(True)
#     plt.legend(y1legend, y2legend, loc='best')
#     plt.show()


def load_data(fname):
    """
  Write code to read the data and return it as 2 numpy arrays.
  Make sure to convert labels to one hot encoded format.
  """
    with open(fname, 'rb') as f:
        file_data = pickle.load(f, encoding='bytes')
    images, labels = file_data[:, :-1], file_data[:, -1]
    labels = labels.reshape(-1, 1)
    return images, labels


class Activation:
    def __init__(self, activation_type="sigmoid"):
        self.activation_type = activation_type
        self.x = None
        # Save the input 'x' for sigmoid or tanh or ReLU to this variable since it will be used later for computing
        # gradients.

    def forward_pass(self, a):
        self.x = a
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.relu(a)

    def backward_pass(self, delta):
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()


        elif self.activation_type == "ReLU":
            grad = self.grad_relu()

        return grad * delta  ##100*101

    def sigmoid(self, x):
        """
    Write the code for sigmoid activation function that takes in a numpy array and returns a numpy array.
    """
        output = 1 / (1 + np.exp(-self.x))
        return output

    def tanh(self, x):
        """
    Write the code for tanh activation function that takes in a numpy array and returns a numpy array.
    """

        output = np.tanh(x)
        return output

    def relu(self, x):
        """
    Write the code for ReLU activation function that takes in a numpy array and returns a numpy array.
    """

        output = np.maximum(self.x, 0)
        return output

    def grad_sigmoid(self):
        """
    Write the code for gradient through sigmoid activation function that takes in a numpy array and returns a numpy array.
    """

        # grad=lambda z: np.multiply(self.sigmoid(self.x), self.sigmoid(-self.x))
        grad = self.sigmoid(self.x) * (1 - self.sigmoid(self.x))
        return grad

    def grad_tanh(self):
        """
    Write the code for gradient through tanh activation function that takes in a numpy array and returns a numpy array.
    """
        tanh_result = np.tanh(self.x)
        grad = 1 - (self.tanh(self.x)) ** 2
        return grad

    def grad_relu(self):
        """
    Write the code for gradient through ReLU activation function that takes in a numpy array and returns a numpy array.
    """
        grad = np.greater(self.x, 0).astype(dtype=int)
        return grad


class Layer:
    def __init__(self, in_units, out_units):
        np.random.seed(42)
        self.w = np.random.randn(in_units, out_units)  # Weight matrix
        self.b = np.zeros((1, out_units)).astype(np.float32)  # Bias
        # self.w=self.w.append(self.w,axis=1)
        self.x = None  # Save the input to forward_pass in this n*input(p)
        self.a = None  # Save the output of forward pass in this (without activation)
        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this
        self.v_w = np.zeros(self.w.shape)
        self.v_b = np.zeros(self.b.shape)

    def forward_pass(self, x):
        """
    Write the code for forward pass through a layer. Do not apply activation function here.
    """
        # x = add_bias_column(x) ##1000*785
        self.x = x  # examples*output
        # print("shape x in ff l",x.shape)
        self.a = self.x @ self.w + self.b  ##n*output ##1000*101
        return self.a

    def backward_pass(self, delta):
        """
    Write the code for backward pass. This takes in gradient from its next layer as input,
    computes gradient for its weights and the delta to pass to its previous layers.
    """
        # print("W shape",self.w.shape)
        # print("x shape",self.x.shape)
        s1 = delta @ self.w.T  ##n*input(j)
        self.d_x = s1
        self.d_w = self.x.T @ delta
        # + (config['L2_penalty']/self.x.shape[0])*self.w  ###input+1*output same as w
        self.d_b = np.ones((self.x.shape[0], 1)).T @ delta

        return self.d_x


class Neuralnetwork:
    def __init__(self, config):
        self.layers = []
        self.x = None  # Save the input to forward_pass in this
        self.y = None  # Save the output vector of model in this
        self.targets = None  # Save the targets in forward_pass in this variable
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def forward_pass(self, x, targets=None):
        """
    Write the code for forward pass through all layers of the model and return loss and predictions.
    If targets == None, loss should be None. If not, then return the loss computed.
    """
        self.x = x.copy()
        for layerl in self.layers:
            x = layerl.forward_pass(x)
        self.y = softmax(x)  ##x 1000*10 y=1000*10
        self.targets = targets
        if targets is not None:
            # loss = self.loss_func(self.y, self.targets)
            loss = self.loss_func(self.y, targets)
        else:
            loss = None
        return loss, self.y

    ###todo targets logits
    def loss_func(self, logits, targets):
        '''
    find cross entropy loss between logits and targets
    '''
        n = logits.shape[0]
        eps = 1e-12
        logits = np.clip(logits, eps, 1 - eps)
        # check
        output = -np.sum(targets * np.log(logits)) / n
        # output = -np.trace(np.dot(targets,logits.T))
        # output = output / len(logits)
        return output

    def backward_pass(self):
        '''
    implement the backward pass for the whole network.
    hint - use previously built functions.
    '''
        # neural net
        delta = self.y - self.targets
        for layerl in reversed(self.layers):
            delta = layerl.backward_pass(delta)

    def update_weights(self):
        for layerl in self.layers:
            if type(layerl) == Layer:
                ##check changed + to -
                ##heck /n
                layerl.v_w = (config['momentum'] * config['momentum_gamma']) * layerl.v_w - (
                            config['learning_rate'] * layerl.d_w)
                # / len(layerl.x)
                layerl.w += layerl.v_w
                layerl.v_b = (config['momentum'] * config['momentum_gamma']) * layerl.v_b + (
                            config['learning_rate'] * layerl.d_b)
                # / len(layerl.x)
                layerl.b += layerl.v_b


def trainer(model, X_train, y_train, X_valid, y_valid, config):
    """
  Write the code to train the network. Use values from config to set parameters
  such as L2 penalty, number of epochs, momentum, etc.
  """
    batch_size = config['batch_size']  # 1000
    num_batches = len(X_train) / batch_size  ##50
    print(y_train.shape)
    y_batches_list = np.array(np.split(y_train, num_batches))  ##50*1000

    epochs_list = []
    loss_list = []
    early_stop = config['early_stop']
    early_stop_count = config['early_stop_epoch']
    last_best_models_list = []

    train_loss_list = []
    validation_loss_list = []
    validation_accuracy_list = []
    train_accurracy_list = []

    prev_loss = 0
    validation_loss = 0
    train_loss = 0
    validation_accuracy = 0
    train_accuracy = 0

    for epoch in range(config['epochs']):
        #         print("EPOCH####", epoch)

        ##prepare result lists will add this in dataframe
        epochs_list.append(epoch)
        train_loss_list.append(train_loss)
        validation_loss_list.append(validation_loss)
        validation_accuracy_list.append(validation_accuracy)
        train_accurracy_list.append(train_accuracy)

        ##early stop condition
        if (early_stop):
            if (validation_loss < prev_loss):
                last_best_models_list.clear()
                tuple_epoch_model = (epoch, copy.deepcopy(model))
                last_best_models_list.append(tuple_epoch_model)
            elif (validation_loss > prev_loss and len(last_best_models_list) < early_stop_count):
                tuple_epoch_model = (epoch, copy.deepcopy(model))
                last_best_models_list.append(tuple_epoch_model)

            prev_loss = validation_loss

        for batch_num, batch in enumerate(np.array(np.split(X_train, num_batches))):
            train_loss, final_y = model.forward_pass(batch, one_hot_encoder(y_batches_list[batch_num]))
            model.backward_pass()
            model.update_weights()

            # loss, final_y = model.forward_pass(batch, y_batches_list[batch_num])
            # print(loss)

        # if X_valid is None:
        #     continue
        # after every epoch, we will measure the accuracy on validation set
        validation_accuracy = test(model, X_valid, y_valid, config)
        train_accuracy = test(model, X_train, y_train, config)

        validation_loss, final_y = model.forward_pass(X_valid, one_hot_encoder(y_valid))

        print('\nEpoch: {},  Validation Accuracy: {} Train Accuracy: {}'.format(epoch, validation_accuracy,
                                                                                train_accuracy))

    df = pd.DataFrame(columns=['EPOCHS', 'TRAIN_LOSS', 'VALIDATION_LOSS', 'VALIDATION_ACCURACY', 'TRAIN_ACCURACY'])

    #     print(len(epochs_list),len(l))
    df['EPOCHS'] = epochs_list
    df['TRAIN_LOSS'] = train_loss_list
    df['VALIDATION_LOSS'] = validation_loss_list
    df['VALIDATION_ACCURACY'] = validation_accuracy_list
    df['TRAIN_ACCURACY'] = train_accurracy_list

    return df, last_best_models_list


# def test(model, X_test, y_test, config):
#   """
#   Write code to run the model on the data passed as input and return accuracy.
#   """
#   batch_size = config['batch_size']
#   N = X_test.shape[0]
#   total_correct = 0
#   for start_idx in range(0, N, batch_size):
#     X_batch = X_test[start_idx:start_idx+batch_size, :]
#     y_batch = y_test[start_idx:start_idx+batch_size, :]
#     # compute forward pass over the mini-batch
#     [loss, predictions] = model.forward_pass(X_batch)
#     # print(predictions)
#     # find the predicted digits
#     predicted_digits = np.argmax(predictions, axis=1)
#     # true digits
#     true_digits = np.argmax(y_batch, axis=1)
#     # print(predicted_digits)
#     # print(true_digits)
#     # correct predictions
#     correct = np.sum(predicted_digits == true_digits)
#     total_correct += correct
#   # finally accuracy
#   accuracy = total_correct / N
#   return accuracy


def test(model, X_test, y_test, config):
    """
  Write code to run the model on the data passed as input and return accuracy.
  """
    loss, predictions = model.forward_pass(X_test, y_test)
    accuracy = 0
    target = one_hot_encoder(y_test)
    for i in range(y_test.shape[0]):
        if np.argmax(predictions[i]) == np.argmax(target[i]):
            accuracy += 1
    result = accuracy
    accuracy = (accuracy) / y_test.shape[0]
    #     print("ACCURRACY", accuracy)
    return accuracy * 100


if __name__ == "__main__":
    train_data_fname = './data/MNIST_train.pkl'
    valid_data_fname = './data/MNIST_valid.pkl'
    test_data_fname = './data//MNIST_test.pkl'

    ### Train the network ###
    model = Neuralnetwork(config)
    X_train, y_train = load_data(train_data_fname)
    X_valid, y_valid = load_data(valid_data_fname)
    X_test, y_test = load_data(test_data_fname)
    df, last_best_models_list = trainer(model, X_train, y_train, X_valid, y_valid, config)

    print(last_best_models_list)
    # last_best_models_list is a tuple epoh versus last best say 5 models
    test_acc = test(last_best_models_list[-1][1], X_test, y_test, config)
    print(test_acc)
    print(df)
