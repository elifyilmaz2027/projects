
import numpy as np
import matplotlib.pyplot as plt
from Network import initialize_parameters
from Network import one_hot
from Network import accuracy
from Network import cross_entropy_loss
from Network import forward_propagation
from Network import backward_propagation
import pickle 


train_inputs = np.load('data/train_inputs.npy')
train_targets = np.load('data/train_targets.npy')
val_inputs = np.load('data/valid_inputs.npy')
val_targets = np.load('data/valid_targets.npy')
vocab = np.load('data/vocab.npy')

#Here, use our vocabulary to get one hot vectors
integer_encoded = []
for i in vocab:
    v = np.where( np.array(vocab) == i)[0][0]
    integer_encoded.append(v)
integer_encoded = np.array(integer_encoded)



one_hot_train_inputs, one_hot_train_targets = one_hot(train_inputs, train_targets,integer_encoded)
one_hot_train_i = np.sum(one_hot_train_inputs, axis=1)
one_hot_val_inputs, one_hot_val_targets = one_hot(val_inputs, val_targets, integer_encoded)
one_hot_val_i = np.sum(one_hot_val_inputs, axis=1)

length = len(one_hot_train_i)
np.random.seed(42)
idx = np.random.permutation(length)
one_hot_train_i, one_hot_train_targets = one_hot_train_i[idx], one_hot_train_targets[idx]


parameters = initialize_parameters()
train_loss_history = []
train_acc_history = []
val_loss_history = []
train_accs = []
train_losses = []
epochs = 60
mini_batch_size = 100
iterations = int(one_hot_train_inputs.shape[0] / mini_batch_size)
for epoch in range(epochs):
    for i in range(0, iterations):
        i = i * mini_batch_size
        lr = 0.1 * (0.5 ** epoch)
        train_results, train_predicted_outputs = forward_propagation(parameters, one_hot_train_i[i:i+mini_batch_size])
        train_loss = cross_entropy_loss(train_predicted_outputs, one_hot_train_targets[i:i+mini_batch_size])
        train_acc = accuracy(train_predicted_outputs, one_hot_train_targets[i:i+mini_batch_size])
        parameters = backward_propagation(train_results, train_predicted_outputs, parameters, one_hot_train_targets[i:i+mini_batch_size], lr)
        train_accs.append(train_acc)
        train_losses.append(train_loss)
    
    val_results, val_predicted_outputs = forward_propagation(parameters, one_hot_val_i)
    val_loss = cross_entropy_loss(val_predicted_outputs, one_hot_val_targets)
    val_acc = accuracy(val_predicted_outputs, one_hot_val_targets)
    val_loss_history.append(val_loss)
    train_loss_history.append(np.array(train_losses).mean())
    train_acc_history.append(np.array(train_accs).mean())
    
    if epoch % 5 == 0:
        print("Epoch: {0}\{1} ,Train_loss: {2}, Train_acc: {3}, Val_loss: {4}, Val_acc: {5}".format(epoch, epochs, np.array(train_losses).mean(), np.array(train_accs).mean(), val_loss, val_acc))
        
#plot history of losses and accuracies to see how to change them by epochs
#train loss change
plt.plot(train_loss_history,'-o')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend(['Train'])
plt.title('Train Loss Change')
plt.show()

#validation loss change
plt.plot(val_loss_history,'-o', color='m')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend(['Validation'])
plt.title('Validation Loss Change')
plt.show()

#train accuracy change
plt.plot(train_acc_history,'-o', color='g')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.legend(['Train'])
plt.title('Train Accuracy Change')
plt.show()

model = {'parameters': parameters, 'train_loss_history': train_loss_history, 'val_loss_history': val_loss_history}
# save the model to disk
filename = 'model.pk'
pickle.dump(model, open(filename, 'wb'))



    

        
        