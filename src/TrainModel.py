#!/usr/bin/python3
import numpy as np
import csv
import time
import shutil
from keras.activations import relu, linear
from keras.models import Sequential
from keras.layers import Dense, Dropout
from talos.model import early_stopper
from talos import Scan
from matplotlib import pyplot as plt


class modelTrainer:
    def __init__(self, mode, plot=False, verbosity=1):
        self.mode = mode
        self.plot = plot
        self.history = None
        self.best_model = None
        self.test_proportion = 0.20
        self.verbosity = verbosity
        if self.mode == 'DoS':
            self.input_dim = 21
            self.dataset_path = 'training_datasets/DoS_dataset_1.csv'
            self.params = {'activation1': [relu],
                          'activation2': [relu],
                          'optimizer': ['Nadam'],
                          'losses': ['mean_absolute_error'],
                          'first_hidden_layer': [20],
                          'second_hidden_layer': [20],
                          'dropout_probability': [0.2],
                          'batch_size': [8],
                          'epochs': [900]}
        elif self.mode == 'Elongation':
            self.input_dim = 26
            self.dataset_path = 'training_datasets/Mechanical_dataset_1.csv'
            self.params = {'activation1': [relu],
                          'activation2': [relu],
                          'optimizer': ['Nadam'],
                          'losses': ['mean_absolute_error'],
                          'first_hidden_layer': [25],
                          'second_hidden_layer': [25],
                          'dropout_probability': [0.2],
                          'batch_size': [8],
                          'epochs': [200]}
        elif self.mode == 'Tensile':
            self.input_dim = 26
            self.dataset_path = 'training_datasets/Mechanical_dataset_1.csv'
            self.params = {'activation1': [relu],
                          'activation2': [relu],
                          'optimizer': ['Nadam'],
                          'losses': ['mean_absolute_error'],
                          'first_hidden_layer': [25],
                          'second_hidden_layer': [25],
                          'dropout_probability': [0.2],
                          'batch_size': [8],
                          'epochs': [30]}
        elif self.mode == 'Yield':
            self.input_dim = 26
            self.dataset_path = 'training_datasets/Mechanical_dataset_1.csv'
            self.params = {'activation1': [relu],
                          'activation2': [relu],
                          'optimizer': ['Nadam'],
                          'losses': ['mean_absolute_error'],
                          'first_hidden_layer': [30],
                          'second_hidden_layer': [30],
                          'dropout_probability': [0.2],
                          'batch_size': [8],
                          'epochs': [60]}
        else:
            return
        self.dataset = {}
        self.loadDataset(self.dataset_path)
        self.run()

    def loadDataset(self, dataset_path):
        dataset = []
        with open(dataset_path, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader)
            for lines in csv_reader:
                dataset.append(lines)
        dataset = np.asarray(dataset)
        np.random.seed(2349138)
        np.random.shuffle(dataset)

        if self.mode == 'Elongation':
            dataset = np.delete(dataset, self.input_dim+2, 1)
            dataset = np.delete(dataset, self.input_dim+1, 1)
        elif self.mode == 'Tensile':
            dataset = np.delete(dataset, self.input_dim+2, 1)
            dataset = np.delete(dataset, self.input_dim, 1)
        elif self.mode == 'Yield':
            dataset = np.delete(dataset, self.input_dim+1, 1)
            dataset = np.delete(dataset, self.input_dim, 1)

        for row in reversed(range(dataset.shape[0])):
            try:
                for column in range((dataset.shape[1])):
                    dataset[row][column] = float(dataset[row][column])
            except Exception:
                dataset = np.delete(dataset, row, 0)
        dataset = dataset.astype(np.float)

        self.dataset['full'] = dataset
        test_split = round(self.dataset['full'].shape[0]*self.test_proportion)
        self.dataset['testing'] = dataset[0:test_split, :]
        self.dataset['training'] = dataset[test_split:self.dataset['full'].shape[0], :]

    def activateTestSplit(self):
        self.dataset['X_train'] = self.dataset['training'][:, 0:self.input_dim]
        self.dataset['Y_train'] = self.dataset['training'][:, self.input_dim]
        self.dataset['X_val'] = self.dataset['testing'][:, 0:self.input_dim]
        self.dataset['Y_val'] = self.dataset['testing'][:, self.input_dim]

    def model(self, X_train, Y_train, X_val, Y_val, params):
        model = Sequential()
        model.add(Dense(params['first_hidden_layer'],
                        input_dim=self.input_dim,
                        activation=params['activation1'],
                        use_bias=True))
        model.add(Dropout(params['dropout_probability']))
        model.add(Dense(params['second_hidden_layer'],
                        activation=params['activation2'],
                        use_bias=True))
        model.add(Dropout(params['dropout_probability']))
        model.add(Dense(1, activation=linear))

        model.compile(optimizer=params['optimizer'],
                      loss=params['losses'])

        history = model.fit(self.dataset['X_train'],
                            self.dataset['Y_train'],
                            batch_size=params['batch_size'],
                            epochs=params['epochs'],
                            verbose=self.verbosity,
                            validation_data=[self.dataset['X_val'], self.dataset['Y_val']]
                            )
        self.history = history
        return history, model

    def run(self):
        self.activateTestSplit()

        self.best_model = Scan(self.dataset['X_train'],
                            self.dataset['Y_train'],
                            model=self.model,
                            params=self.params,
                            print_params=True,
                            experiment_name=self.mode + '_model',
                            reduction_metric='val_loss').best_model(metric='val_loss', asc=True)

        if self.plot:
            plt.plot(np.log(self.history.history['loss'])/np.log(40))
            plt.plot(np.log(self.history.history['val_loss'])/np.log(40))
            plt.title(self.mode + ' Training Results')
            plt.ylabel('Log(40) Linear Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.show()

        print('Model trained on ', self.dataset['X_train'].shape[0], ' samples, verified on ',
              self.dataset['X_val'].shape[0], ' samples')

        shutil.rmtree(self.mode + '_model')

        model_path = 'models/' + self.mode + '_model' + '_' + str(int(round(time.time() * 1000))) + '.h5'
        self.best_model.save(model_path)
        print('Model saved to: ' + model_path)
