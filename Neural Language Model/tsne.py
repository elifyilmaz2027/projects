import numpy as np
import pickle

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

vocab = np.load('data/vocab.npy')
integer_encoded = []
for i in vocab:
    v = np.where( np.array(vocab) == i)[0][0]
    integer_encoded.append(v)
integer_encoded = np.array(integer_encoded)

shape = (integer_encoded.size, integer_encoded.max()+1)
one_hot_integer_encoded = np.zeros(shape)
rows= np.arange(integer_encoded.size)
one_hot_integer_encoded[rows, integer_encoded] = 1

tsne = TSNE(n_components=2, verbose=1)

'''at first, without using word embeddings, let see 
how to look like similarity of our words.'''

t = tsne.fit_transform(one_hot_integer_encoded)
plt.figure(figsize=(10, 10))
plt.title('Similarity of Words Without Using Embeddings')
for i, txt in enumerate(vocab):
    plt.scatter(t[i,0], t[i,1])
    plt.annotate(txt, (t[i,0], t[i,1]))
plt.show()

model = pickle.load(open('model.pk', 'rb'))
parameters = model['parameters']

w1,w2,w3,b1,b2 = parameters

'''while creating word embeddings, because of concatenating 250x16 dimensional
w1's, we have 48 dimensional w1 but we use its 16 dimensions.'''
w11 = w1[:,:16]
w12 = w1[:,16:32]
w13 = w1[:,32:48]


a1 = one_hot_integer_encoded @ w11
a2 = one_hot_integer_encoded @ w12
a3 = one_hot_integer_encoded @ w13



t1 = tsne.fit_transform(a1)
t2 = tsne.fit_transform(a2) 
t3 = tsne.fit_transform(a3) 

plt.figure(figsize=(10, 10))
plt.title('Similarity of Words 1')
for i, txt in enumerate(vocab):
    plt.scatter(t1[i,0], t1[i,1])
    plt.annotate(txt, (t1[i,0], t1[i,1]))
plt.show()

plt.figure(figsize=(10, 10))
plt.title('Similarity of Words 2')
for i, txt in enumerate(vocab):
    plt.scatter(t2[i,0], t2[i,1])
    plt.annotate(txt, (t2[i,0], t2[i,1]))
plt.show()

plt.figure(figsize=(10, 10))
plt.title('Similarity of Words 3')
for i, txt in enumerate(vocab):
    plt.scatter(t3[i,0], t3[i,1])
    plt.annotate(txt, (t3[i,0], t3[i,1]))
plt.show()