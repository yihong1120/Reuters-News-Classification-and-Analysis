import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils.np_utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class ReutersModel:
    def __init__(self, num_words:int=8000, input_dim:int=8000) -> None:
        """
        Initialise the ReutersModel class.

        Args:
            num_words (int): Number of most frequent words to consider.
            input_dim (int): Input dimension for the neural network.
        """
        self.num_words = num_words
        self.input_dim = input_dim
        self.model = self.build_model()
        self.word_index = None
        self.reverse_word_index = None

    def build_model(self) -> tf.keras.models.Model:
        """
        Build the Reuters neural network model.

        Returns:
            keras.models.Model: Compiled neural network model.
        """
        model = models.Sequential()
        model.add(Dense(64, activation='relu', input_shape=(self.input_dim,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(46, activation='softmax'))
        model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def load_data(self):
        """
        Load the Reuters dataset.

        Returns:
            Tuple: Training data, training labels, test data, test labels.
        """
        return reuters.load_data(num_words=self.num_words)

    def preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess the provided data.

        Args:
            data (np.ndarray): Data to preprocess.

        Returns:
            np.ndarray: Preprocessed data.
        """
        return np.array([self.vectorize_sequence(sequence) for sequence in data])

    def vectorize_sequence(self, sequence: list) -> np.ndarray:
        """
        Vectorise a given sequence.

        Args:
            sequence (list): List of word indices.

        Returns:
            np.ndarray: Vectorised sequence.
        """
        result = np.zeros(self.input_dim)
        for index in sequence:
            result[index] = 1
        return result

    def one_hot_encode_labels(self, labels: list) -> np.ndarray:
        """
        One-hot encode the given labels.

        Args:
            labels (list): Labels to encode.

        Returns:
            np.ndarray: One-hot encoded labels.
        """
        return to_categorical(labels)

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, epochs:int=10, batch_size:int=256) -> tf.python.keras.callbacks.History:
        """
        Train the model.

        Args:
            x_train (np.ndarray): Training data.
            y_train (np.ndarray): Training labels.
            x_val (np.ndarray): Validation data.
            y_val (np.ndarray): Validation labels.
            epochs (int): Number of epochs for training.
            batch_size (int): Batch size for training.

        Returns:
            tf.python.keras.callbacks.History: Training history.
        """
        return self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> list:
        """
        Evaluate the model.

        Args:
            x_test (np.ndarray): Test data.
            y_test (np.ndarray): Test labels.

        Returns:
            list: Evaluation results.
        """
        return self.model.evaluate(x_test, y_test)

    def predict(self, text: str) -> np.ndarray:
        """
        Predict the category for a given text.

        Args:
            text (str): Text to predict.

        Returns:
            np.ndarray: Model predictions.
        """
        if not self.word_index:
            self.word_index = reuters.get_word_index()
            self.reverse_word_index = dict([(value, key) for (key, value) in self.word_index.items()])
        text_vector = self.preprocess_data([text])
        return self.model.predict(text_vector)

    def save(self, model_dir:str='reuters_model.h5') -> None:
        """
        Save the model to a file.

        Args:
            model_dir (str): Path to save the model.
        """
        self.model.save(model_dir)

    def load(self, model_dir:str='reuters_model.h5') -> None:
        """
        Load the model from a file.

        Args:
            model_dir (str): Path to load the model from.
        """
        self.model = tf.keras.models.load_model(model_dir)

class ReutersTrainer:
    def __init__(self, reuters_model: ReutersModel) -> None:
        """
        Initialise the ReutersTrainer class.

        Args:
            reuters_model (ReutersModel): Instance of the ReutersModel class.
        """
        self.reuters_model = reuters_model

    def train_and_evaluate(self) -> (tf.python.keras.callbacks.History, list):
        """
        Train and evaluate the model.

        Returns:
            tf.python.keras.callbacks.History: Training history.
            list: Evaluation results.
        """
        train_data, train_labels, test_data, test_labels = self.reuters_model.load_data()
        x_train = self.reuters_model.preprocess_data(train_data)
        x_test = self.reuters_model.preprocess_data(test_data)
        y_train = self.reuters_model.one_hot_encode_labels(train_labels)
        y_test = self.reuters_model.one_hot_encode_labels(test_labels)

        x_val = x_train[:1000]
        partial_x_train = x_train[1000:]
        y_val = y_train[:1000]
        partial_y_train = y_train[1000:]

        history = self.reuters_model.train(partial_x_train, partial_y_train, x_val, y_val)
        results = self.reuters_model.evaluate(x_test, y_test)
        return history, results

class ReutersPredictor:
    def __init__(self, reuters_model: ReutersModel) -> None:
        """
        Initialise the ReutersPredictor class.

        Args:
            reuters_model (ReutersModel): Instance of the ReutersModel class.
        """
        self.reuters_model = reuters_model
        # Define the list of labels/categories
        self.labels = ['copper', 'cocoa', 'sugar', 'gold', 'iron-steel', 'tin', 'soybean', 'oilseed', 'coffee', 'livestock', 'wheat', 'alum', 'rubber', 'veg-oil', 'palm-oil', 'housing', 'nat-gas', 'money-fx', 'heat', 'ship', 'orange', 'grain', 'wpi', 'carcass', 'retail', 'potato', 'crude', 'fuel', 'pet-chem', 'strategic-metal', 'lead', 'lei', 'interest', 'zinc', 'income', 'reserves', 'dlr', 'corn', 'gnp', 'meal-feed', 'bop', 'cpu', 'money-supply', 'gnp-def']

    def predict(self, text: str) -> str:
        """
        Predict the label/category for a given text.

        Args:
            text (str): Text to predict.

        Returns:
            str: Predicted label/category.
        """
        predictions = self.reuters_model.predict(text)
        predicted_label = self.labels[np.argmax(predictions)]
        return predicted_label

if __name__ == "__main__":
    # Initialise the Reuters model
    reuters_model = ReutersModel()
    # Train and evaluate the model
    trainer = ReutersTrainer(reuters_model)
    history, results = trainer.train_and_evaluate()
    print('Results:', results)

    # Predict the label for a new text
    reuters_predictor = ReutersPredictor(reuters_model)
    new_text = "Japan's Fujitsu Ltd said it will begin a pilot field trial of a subscription-based digital farming service in Australia from mid-February, using its Akisai agricultural IT platform."
    predicted_label = reuters_predictor.predict(new_text)
    print('Predicted label:', predicted_label)

    # Save the trained model
    reuters_model.save()
