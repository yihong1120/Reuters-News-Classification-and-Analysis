import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils.np_utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ReutersModel:
    def __init__(self, num_words=8000, input_dim=8000):
        self.num_words = num_words
        self.input_dim = input_dim
        self.model = self.build_model()
        self.word_index = None
        self.reverse_word_index = None

    def build_model(self):
        model = models.Sequential()
        model.add(Dense(64, activation='relu', input_shape=(self.input_dim,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(46, activation='softmax'))
        model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def load_data(self):
        (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=self.num_words)
        return train_data, train_labels, test_data, test_labels

    def preprocess_data(self, data):
        return np.array([self.vectorize_sequence(sequence) for sequence in data])

    def vectorize_sequence(self, sequence):
        result = np.zeros(self.input_dim)
        for index in sequence:
            result[index] = 1
        return result

    def one_hot_encode_labels(self, labels):
        return to_categorical(labels)

    def train(self, x_train, y_train, x_val, y_val, epochs=10, batch_size=256):
        history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))
        return history

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predict(self, text):
        if not self.word_index:
            self.word_index = reuters.get_word_index()
            self.reverse_word_index = dict([(value, key) for (key, value) in self.word_index.items()])

        text_vector = self.preprocess_data([text])
        return self.model.predict(text_vector)

    def save(self, model_dir='reuters_model.h5'):
        self.model.save(model_dir)

    def load(self, model_dir='reuters_model.h5'):
        self.model = tf.keras.models.load_model(model_dir)

class ReutersTrainer:
    def __init__(self, reuters_model):
        self.reuters_model = reuters_model

    def train_and_evaluate(self):
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
    def __init__(self, reuters_model):
        self.reuters_model = reuters_model
        self.labels = ['copper', 'cocoa', 'sugar', 'gold', 'iron-steel', 'tin', 'soybean', 'oilseed', 'coffee', 'livestock', 'wheat', 'alum', 'rubber', 'veg-oil', 'palm-oil', 'housing', 'nat-gas', 'money-fx', 'heat', 'ship', 'orange', 'grain', 'wpi', 'carcass', 'retail', 'potato', 'crude', 'fuel', 'pet-chem', 'strategic-metal', 'lead', 'lei', 'interest', 'zinc', 'income', 'reserves', 'dlr', 'corn', 'gnp', 'meal-feed', 'bop', 'cpu', 'money-supply', 'gnp-def']

    def predict(self, text):
        predictions = self.reuters_model.predict(text)
        predicted_label = self.labels[np.argmax(predictions)]
        return predicted_label

# Usage
if __name__ == "__main__"
    reuters_model = ReutersModel()
    trainer = ReutersTrainer(reuters_model)
    history, results = trainer.train_and_evaluate()

    print('Results:', results)

    reuters_predictor = ReutersPredictor(reuters_model)
    new_text = "Japan's Fujitsu Ltd said it will begin a pilot field trial of a subscription-based digital farming service in Australia from mid-February, using its Akisai agricultural IT platform."
    predicted_label = reuters_predictor.predict(new_text)

    print('Predicted label:', predicted_label)

    reuters_model.save()
