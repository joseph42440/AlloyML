#!/usr/bin/python3

import numpy as np
import csv

class predictOutput:
    def __init__(self, dataset_path, models, mode):
        self.mode = mode
        self.models = models
        self.dataset_path = dataset_path
        self.dataset = []
        with open(self.dataset_path, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader)
            for lines in csv_reader:
                self.dataset.append(lines)
        self.dataset = np.asarray(self.dataset)

    def predict(self, index, datapoint):
        if self.mode == 'DoS':
            prediction = round(self.models['DoS'].predict(np.reshape(datapoint, (1, -1)))[0][0], 2)
            print(str(index) + ': DoS=' + str(prediction))
        elif self.mode == 'Mechanical':
            prediction_1 = round(self.models['elongation'].predict(np.reshape(datapoint, (1, -1)))[0][0], 2)
            prediction_2 = round(self.models['tensile'].predict(np.reshape(datapoint, (1, -1)))[0][0], 2)
            prediction_3 = round(self.models['yield'].predict(np.reshape(datapoint, (1, -1)))[0][0], 2)

            print(str(index) + ': elongation%=' + str(prediction_1) + ' tensile(MPa)=' + str(prediction_2) +
                  ' yield(MPa)=' + str(prediction_3))

    def run(self):
        for i in range(self.dataset.shape[0]):
            self.predict(i, self.dataset[i])
