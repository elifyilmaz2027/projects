
###### Forward Propagation ######

import numpy as np

#define sigmoid function
def sigmoid(inputs):
    return 1 / (1 + np.exp(- inputs))


#define softmax function
def softmax(inputs):
    exponentials = np.exp(inputs - np.max(inputs))
    return exponentials / np.sum(exponentials + 10e-08)



#initialize parameters
def initialize_parameters():
    np.random.seed(42)
    w_11 = np.random.randn(250,16) * 10e-04
    w_1 = np.concatenate((w_11, w_11, w_11), axis=1)
    np.random.seed(42)
    w_21 = np.random.randn(16,128) * 10e-04
    np.random.seed(43)
    w_22 = np.random.randn(16,128) * 10e-04
    np.random.seed(44)
    w_23 = np.random.randn(16,128) * 10e-04
    w_2 = np.concatenate((w_21, w_22, w_23), axis=0)
    np.random.seed(42)
    w_3 = np.random.randn(128,250) * 10e-04
    b_1 = np.zeros((1, 128))
    b_2 = np.zeros((1, 250))
    return w_1, w_2, w_3, b_1, b_2



#apply forward propagation
def forward_propagation(parameters, data):
    l, m = data.shape
    w_1, w_2, w_3, b_1, b_2 = parameters
    z_1 = data @ w_1
    z_2 = z_1 @ w_2 + b_1
    z_3 = []
    for i in range(0, l):
        x = sigmoid(z_2[i])
        z_3.append(x)
    z_3 = np.array(z_3)
    z_3 = sigmoid(z_2)
    z_4 = z_3 @ w_3 + b_2
    predicted_outputs = []
    for i in range(0, l):
        predicted_output = softmax(z_4[i])
        predicted_outputs.append(predicted_output)
    predicted_outputs = np.array(predicted_outputs)
    results =  data, z_1, z_2, z_3, z_4
    return results, predicted_outputs

###### Backward Propagation ######

#define gradient softmax and sigmoid functions
def gradient_softmax(inputs):
    l = len(inputs)
    softmax_outputs = []
    for i in range(0, l):
        softmax_output = softmax(inputs[i])
        softmax_outputs.append(softmax_output)
    softmax_outputs = np.array(softmax_outputs)
    d = np.zeros(softmax_outputs.shape)
    np.fill_diagonal(d, 1, wrap=True)
    gradient = softmax_outputs * (d - softmax_outputs)
    return gradient  

def gradient_sigmoid(inputs):
    l = len(inputs)
    a = []
    for i in range(0, l):
        x = sigmoid(inputs[i])
        a.append(x)
    a = np.array(a)
    return a * (1 - a)

#apply bacward propagation algorithm
def backward_propagation(results, output, parameters, one_hot_targets, learning_rate):

    w_1, w_2, w_3, b_1, b_2 = parameters
    data, z_1, z_2, z_3, z_4 = results
    length = len(data)
    
    ##compute gradients 
    #start with gradient of loss with respect to our predicted values
    d_wrt_output = (1 / length) * np.divide(one_hot_targets, output + 10e-08)
    d_wrt_z4 = d_wrt_output * gradient_softmax(z_4)
    d_wrt_w3 = (1 / length) * z_3.T @ d_wrt_z4
    d_wrt_b2 = np.mean(d_wrt_z4, axis=0, keepdims=True)
    d_wrt_z3 = (1 / length) * d_wrt_z4 @ w_3.T
    d_wrt_z2 = d_wrt_z3 * gradient_sigmoid(z_2)
    d_wrt_w2 = (1 / length) * z_1.T @ d_wrt_z2 
    d_wrt_b1 = np.mean(d_wrt_z2, axis=0, keepdims=True)
    d_wrt_z1 = (1 / length) * d_wrt_z2 @ w_2.T
    d_wrt_w1 = (1 / length) * data.T @ d_wrt_z1

    #update the weights and biases
    W_1_star = w_1 - learning_rate * d_wrt_w1
    W_2_star = w_2 - learning_rate * d_wrt_w2
    W_3_star = w_3 - learning_rate * d_wrt_w3
    b_1_star = b_1 - learning_rate * d_wrt_b1
    b_2_star = b_2 - learning_rate * d_wrt_b2
    
    return W_1_star, W_2_star, W_3_star, b_1_star, b_2_star

#define cross-entropy loss function
def cross_entropy_loss(predicted_outputs, one_hot_actual_targets):
    length = len(one_hot_actual_targets)
    for i in range(0, length):
        loss_ = - one_hot_actual_targets[i] * np.log(10e-08 + predicted_outputs[i])
    loss = np.sum(loss_) / length
    return loss

#define accuracy as a fuction looking for whether or not our predictions actual targets are equal.
def accuracy(predicted_outputs, one_hot_actual_targets):
    length = len(one_hot_actual_targets)
    actual_targets = one_hot_actual_targets.argmax(axis=1)
    predicted_targets = predicted_outputs.argmax(axis=1)
    correct = np.count_nonzero((np.equal(predicted_targets, actual_targets)))
    acc = correct / length
    return acc


#define a one hot function for inputs and targets 
def one_hot(inputs, targets, integer_encoded):
    shape_inputs = (inputs.size, integer_encoded.max()+1)
    one_hot_inputs = np.zeros(shape_inputs)
    rows_inputs = np.arange(inputs.size)
    inputs_re = inputs.reshape(inputs.size)
    one_hot_inputs[rows_inputs, inputs_re] = 1
    one_hot_inputs = one_hot_inputs.reshape(inputs.shape[0], inputs.shape[1], integer_encoded.max()+1)
    
    shape_targets = (targets.size, integer_encoded.max()+1)
    one_hot_targets = np.zeros(shape_targets)
    rows_targets = np.arange(targets.size)
    one_hot_targets[rows_targets, targets] = 1
    
    return one_hot_inputs, one_hot_targets




    
