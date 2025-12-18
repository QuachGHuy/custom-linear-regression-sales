import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, data_X, data_Y, learning_rate, epochs, batch_size):
        self.data_X = data_X
        self.data_Y = data_Y
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.data_size = data_X.shape[0]

        self.theta = np.random.randn(self.data_X.shape[1])
        self.loss = []
        self.accuracy = []

    def compute_mse(self, y, y_pred):
        loss = (y - y_pred) * (y - y_pred)
        loss = np.mean(loss)
        return loss

    def compute_r2_score(self, y, y_pred):
        tss = np.sum((y - np.mean(y)) ** 2)
        rss = np.sum((y - y_pred) ** 2)

        return 1 - (rss / tss)

    def predict(self, X):
        y_pred = np.dot(X, self.theta)
        return y_pred

    def fit(self):
        for epoch in range(self.epochs):
            epoch_loss = []
            epoch_acc = []

            # 1. Shuffle data
            indices = np.random.permutation(self.data_size)
            X_shuffled = self.data_X[indices]
            Y_shuffled = self.data_Y[indices]

            for i in range(0, self.data_size, self.batch_size):
                # 2. Get batch data
                X_batch = X_shuffled[i:i + self.batch_size,:]
                Y_batch = Y_shuffled[i:i + self.batch_size]

                # 3. Predict
                y_pred = self.predict(X_batch)

                # 4. Compute Accuracy
                acc = self.compute_r2_score(Y_batch, y_pred)
                epoch_acc.append(acc)

                # 5. Compute Loss
                loss = self.compute_mse(Y_batch, y_pred)
                epoch_loss.append(loss)

                # 6. Gradient Descent
                k = 2*(y_pred - Y_batch)
                gradient = np.dot(X_batch.T, k) / self.batch_size

                # 7. Update theta
                self.theta = self.theta - self.learning_rate * gradient

            self.loss.append(np.mean(epoch_loss))
            self.accuracy.append(np.mean(epoch_acc))

            print(f"Epoch: {epoch} MSE: {self.loss[-1]}, Acc:{self.accuracy[-1]} \n")

        response = f'Training Complete \n \
                    Theta: {self.theta}, \n \
                    RMSE: {np.sqrt(self.loss[-1])}, \n \
                    Accuracy: {self.accuracy[-1]}'
        print(response)
        return 
        
    def plot(self):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(np.sqrt(self.loss), color="red")
        plt.xlabel("Epochs")
        plt.ylabel("RMSE")

        plt.subplot(1, 2, 2)
        plt.plot(self.accuracy, color="blue")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")

        plt.show()