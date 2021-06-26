import torch
import pickle
import matplotlib.pyplot as plt


model = pickle.load(open('model.pk', 'rb'))
train_reconstruction_images = model['train_reconstruction_images']
train_loss_history = model['train_loss_history']
train_regularization_terms = model['train_regularization_terms']

#Plot Loss Change
plt.plot(train_loss_history,'-o')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend(['Train'])
plt.title('Train Loss Change for VAE Model')
plt.show()

#Plot Regularization Term
plt.plot(train_regularization_terms,'-o')
plt.xlabel('epoch')
plt.ylabel('Regularization Term with KL Divergence')
plt.legend(['Train'])
plt.title('Change of Train Regularization Term with KL Divergence for VAE Model')
plt.show()


#plot some of the train data
figure = plt.figure(figsize=(10, 8))
plt.title('Random Images from Train Reconstruction Images with VAE')
plt.axis("off")
cols, rows = 10, 10
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_reconstruction_images), size=(1,)).item()
    img = train_reconstruction_images[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

