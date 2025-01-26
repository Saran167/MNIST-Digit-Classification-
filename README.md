This project demonstrates the MNIST dataset (handwritten digits) is normalized to a range of [0, 1] to help the neural network train efficiently.

DATASET

MNIST (Modified National Institute of Standards and Technology) is a collection of 70,000 grayscale images of handwritten digits (0–9).
The dataset is divided into:
60,000 training images: Used to train the machine learning model.
10,000 test images: Used to evaluate the model's performance on unseen data.Each image is 28x28 pixels.
The dataset is labeled, meaning each image has a corresponding digit (0-9) as the ground truth.

FEATURES

Training Images: x_train (60,000 samples, each 28x28 pixels).
Test Images: x_test (10,000 samples, each 28x28 pixels).
Labels: y_train and y_test contain the true digit labels (ranging from 0 to 9).

Build the Neural Network Model:

Flatten Layer: Converts the 28x28 images into a 1D array of 784 values (28 * 28 = 784).
Dense Layers: Two fully connected layers:
First hidden layer with 128 neurons and ReLU activation.
Second hidden layer with 64 neurons and ReLU activation.
Output Layer: A softmax layer with 10 neurons (one for each class from 0 to 9), converting the outputs into probabilities.

Compile the Model:

Optimizer: Adam optimizer is used, which is adaptive and works well for deep learning tasks.
Loss Function: Sparse Categorical Crossentropy is used for multi-class classification problems where the labels are integers.
Metrics: Accuracy is used to measure the model's performance.

Train the Model:

Epochs: The model is trained for 5 epochs, where an epoch represents one full pass through the training data.
Batch Size: The training data is divided into mini-batches of 32 samples.
Validation Split: 20% of the training data is used for validation to monitor the model's performance during training.

Evaluate the Model:

After training, the model is evaluated on the test set (x_test, y_test).
The evaluate function computes the test loss and accuracy.

Make Predictions:

The model predicts probabilities for each test image using the predict function.
y_pred_classes: Contains the predicted class for each test image (digit 0–9), determined by selecting the class with the highest probability.

Classification Report:

The classification report provides detailed performance metrics (precision, recall, F1-score) for each digit class (0-9).

OUTPUT

Test Accuracy: This will display the overall accuracy of the model on the test dataset.

Classification Report: This will provide the precision, recall, and F1-score for each class (digit) and show the model’s performance across all classes.

