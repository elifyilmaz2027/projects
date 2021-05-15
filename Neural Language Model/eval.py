
import numpy as np
import pickle
from Network import one_hot
from Network import forward_propagation
from Network import accuracy
from Network import cross_entropy_loss

vocab = np.load('data/vocab.npy')
integer_encoded = []
for i in vocab:
    v = np.where( np.array(vocab) == i)[0][0]
    integer_encoded.append(v)
integer_encoded = np.array(integer_encoded)

test_inputs = np.load('data/test_inputs.npy')
test_targets = np.load('data/test_targets.npy')


model = pickle.load(open('model.pk', 'rb'))
   
parameters = model['parameters']

one_hot_test_inputs, one_hot_test_targets = one_hot(test_inputs, test_targets, integer_encoded)
one_hot_test_i = np.sum(one_hot_test_inputs, axis=1)

test_results, test_predicted_output = forward_propagation(parameters, one_hot_test_i)
test_loss = cross_entropy_loss(test_predicted_output, one_hot_test_targets)
test_acc = accuracy(test_predicted_output, one_hot_test_targets)


print("Test Loss: {0}, Test Accuracy: {1}".format(test_loss, test_acc))

## 'city of new'
index_city = np.where(vocab == 'city')
index_of = np.where(vocab == 'of')
index_new = np.where(vocab == 'new')
city_of_new = np.zeros((1, len(vocab)))
city_of_new[:,index_city] = 1
city_of_new[:,index_of] = 1
city_of_new[:,index_new] = 1
_, predicition_city_of_new = forward_propagation(parameters, city_of_new)
print('city of new', vocab[predicition_city_of_new.argmax()])

## 'life in the'
index_life = np.where(vocab == 'life')
index_in = np.where(vocab == 'in')
index_the = np.where(vocab == 'the')
life_in_the = np.zeros((1, len(vocab)))
life_in_the[:,index_life] = 1
life_in_the[:,index_in] = 1
life_in_the[:,index_the] = 1
_, predicition_life_in_the = forward_propagation(parameters, life_in_the)
print('life in the', vocab[predicition_life_in_the.argmax()])

## 'life in the'
index_he = np.where(vocab == 'he')
index_is = np.where(vocab == 'is')
index_the = np.where(vocab == 'the')
he_is_the = np.zeros((1, len(vocab)))
he_is_the[:,index_he] = 1
he_is_the[:,index_is] = 1
he_is_the[:,index_the] = 1
_, predicition_he_is_the = forward_propagation(parameters, he_is_the)
print('he is the', vocab[predicition_he_is_the.argmax()])
