# preprocessing
from param_graph.generate_nns import generate_random_cnn
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim

from param_graph.seq_to_net import seq_to_net

def generate_data(n_architectures=10, n_epochs=2, lr=0.001, momentum=0.9):
  """
  Generates random CNN architectures and trains them on CIFAR10 data

  Args:
  - n_architectures: int, the number of architectures to generate
  - n_epochs: int, the number of epochs to train each architecture
  - other hyperpa
  """



  # load cifar10
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

  # reshape x to be of the form (batch_size, 3, 32, 32)
  x_train = x_train.transpose(0, 3, 1, 2)
  x_test = x_test.transpose(0, 3, 1, 2)

  # sample some data
  x_train = x_train[:1000]
  y_train = y_train[:1000]
  x_test = x_test[:1000]
  y_test = y_test[:1000]


  results = []

  for i in range(n_architectures):
    cnn = generate_random_cnn(in_dim=32, in_channels=3, out_dim=10)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr=lr, momentum=momentum)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    # Train the model
    for epoch in range(n_epochs):
      running_loss = 0.0
      for i in range(len(x_train_tensor)):
        optimizer.zero_grad()
        outputs = cnn(x_train_tensor[i].unsqueeze(0))
        loss = criterion(outputs, y_train_tensor[i])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
          print(f'Model {i + 1}/{n_architectures}, Epoch {epoch + 1}/){n_epochs}, Batch {i + 1}: Loss {running_loss / 1000}')
          running_loss = 0.0

      # calculate accuracy
      correct = 0
      total = 0
      with torch.no_grad():
        for i in range(len(x_test)):
          outputs = cnn(torch.tensor(x_test, dtype=torch.float32).unsqueeze(0))
          _, predicted = torch.max(outputs.data, 2)
          total += 1
          correct += (predicted == y_test[i]).sum().item()

      accuracy = correct / total

    node_feats, edge_indices, edge_feats = seq_to_net(cnn).get_feature_tensors()
    results.append([node_feats, edge_indices, edge_feats, accuracy])

    # Record results in csv
    import pandas as pd
    df = pd.DataFrame(results, columns=['node_feats', 'edge_indices', 'edge_feats', 'accuracy'])
    df.to_csv('data_cnn.csv')
