# Neural-network-
This project demonstrates how to train a neural network on the Iris dataset using TensorFlow and Keras. The Iris dataset contains 150 samples of iris flowers with 4 features: sepal length, sepal width, petal length, and petal width. The task is to classify these samples into three different species: Setosa, Versicolor, and Virginica.

Procedure

Dataset:
The Iris dataset consists of 150 samples, each with 4 features:

Sepal Length Sepal Width Petal Length Petal Width There are 3 target classes (species):

Setosa Versicolor Virginica

Preprocessing:
One-Hot Encoding: The target labels are one-hot encoded to be compatible with the softmax activation in the output layer.

Feature Scaling: The features are standardized (zero mean and unit variance) to speed up the training process and improve convergence.

Model:
A simple feedforward neural network is constructed with two hidden layers, each using the ReLU activation function.

The output layer employs the softmax activation function to perform multi-class classification.

Training:
The model is trained using the Adam optimizer and categorical crossentropy as the loss function.

Training is performed for 50 epochs with a batch size of 8, and a validation split of 20% is used to monitor the model's performance during training.
Evaluation:
After training, the model is evaluated on a test set to determine its accuracy.

Output:

Training Output:
During training, the loss and accuracy of both the training and validation sets are shown after each epoch:

Epoch 1/50

12/12 ━━━━━━━━━━━━━━━━━━━━ 1s 23ms/step - accuracy: 0.0587 - loss: 1.1380 - val_accuracy: 0.3750 - val_loss: 1.0721

Epoch 2/50

12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - accuracy: 0.2628 - loss: 1.0720 - val_accuracy: 0.5833 - val_loss: 1.0192

...

Epoch 50/50

12/12 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - accuracy: 0.9311 - loss: 0.1997 - val_accuracy: 0.9583 - val_loss: 0.2729

Test Accuracy:
After training, the model is evaluated on the test dataset:

Test Accuracy: 1.00

This indicates that the model achieved 100% accuracy on the test set.

Predictions:
The model makes predictions on the test set, and the predicted classes are compared with the true classes:

Predicted classes: [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0]

True classes: [1 0 2 1 1 0 1 2 1 1 2 0 0 0 0 1 2 1 1 2 0 2 0 2 2 2 2 2 0 0]

Explanation:
Training Output: Displays the loss and accuracy for both training and validation sets after each epoch.

Test Accuracy: The model's accuracy when evaluated on the test dataset (100% accuracy in this case).

Predictions: The predicted classes and the true classes for the test dataset, showing that all predictions match the true classes.
