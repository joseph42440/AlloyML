#!/usr/bin/python3

from copy import deepcopy
import numpy as np
from scipy.stats import truncnorm

class AlDatapoint:
    def __init__(self, categorical_inputs, range_based_inputs):
        self.categorical_inputs = categorical_inputs
        self.range_based_inputs = range_based_inputs

    def formatForInput(self):
        my_input = [*self.categorical_inputs.values(), self.getAl(), *self.range_based_inputs.values()]
        return np.reshape(my_input, (1, -1))

    def print(self):
        for key, value in self.categorical_inputs.items():
            print(key, value)
        print('Al%', round(self.getAl(), 2))
        for key, value in self.range_based_inputs.items():
            if value:
                print(key, value)

    def getAl(self):
        return 100 - sum(self.range_based_inputs.values())

class scanSettings:
    def __init__(self, mode):
        self.mode = mode

        if self.mode == 'DoS':
            self.loss_type = 'Linear'
            self.max_steps = 1000
            self.targets = {
                'DoS': 10
            }
            self.categorical_inputs = {
                'time(days)': 7,
                'temperature(C)': 150,
                'recrystallised': [1],
                'temper': [1, 2, 4, 5]
            }
            self.range_based_inputs = dict(zip(
                ['Ag%', 'Ca%', 'Ce%', 'Cr%', 'Cu%', 'Fe%', 'Ge%', 'Mg%', 'Mn%', 'Nd%', 'Ni%', 'Si%', 'Sr%', 'Ti%', 'Zn%', 'Zr%'],
                [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [4, 5.5], [0, 0],
                 [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]))

        if self.mode == 'Mechanical':
            self.loss_type = 'Percentage'
            self.max_steps = 1000
            self.targets = {
                'elongation%': 6,
                'yield strength(MPa)': 250
            }
            self.categorical_inputs = {
                'processing condition': [1, 2, 4, 6, 9, 10],
            }
            self.range_based_inputs = dict(zip(
                ['Ag%', 'B%', 'Be%', 'Bi%', 'Cd%', 'Co%', 'Cr%', 'Cu%', 'Er%', 'Eu%', 'Fe%', 'Ga%', 'Li%', 'Mg%', 'Mn%',
                 'Ni%', 'Pb%', 'Sc%', 'Si%', 'Sn%', 'Ti%', 'V%', 'Zn%', 'Zr%'],
                [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                 [0, 0], [0, 0], [0, 0], [0, 0], [4, 5.5], [0, 0], [0, 0], [0, 0], [0, 0],
                 [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]))

class compositionScan:
    def __init__(self, settings, models):
        self.step_batch_size = 50
        self.step_final_std = 0.01
        self.finetune_max_rounds = 10
        self.finetune_batch_size = 500
        self.mode = settings.mode
        self.loss_type = settings.loss_type
        self.targets = settings.targets
        self.max_steps = settings.max_steps
        self.categorical_inputs = settings.categorical_inputs
        self.range_based_inputs = settings.range_based_inputs
        self.models = models

    def generateDatapoint(self):
        if self.mode == 'DoS' or self.mode == 'Mechanical':
            return AlDatapoint(self.categorical_inputs, self.range_based_inputs)

    def calculateLoss(self, datapoint):
        if self.mode == 'DoS':
            return abs(self.models['DoS'].predict(datapoint.formatForInput())[0] - self.targets['DoS'])
        elif self.mode == 'Mechanical':
            return ((abs((self.models['elongation'].predict(datapoint.formatForInput())[0]/self.targets['elongation%'])-1)*100 \
                   + abs((self.models['yield'].predict(datapoint.formatForInput())[0]/self.targets['yield strength(MPa)'])-1)*100))/2

    def printResults(self, best_datapoint):
        best_datapoint.print()
        if self.mode == 'DoS':
            print('Results in a predicted %f DoS' % (self.models['DoS'].predict(best_datapoint.formatForInput())[0]))
        elif self.mode == 'Mechanical':
            print('Results in a predicted %f elongation(%%)' % (self.models['elongation'].predict(best_datapoint.formatForInput())[0]))
            print('Results in a predicted %f tensile strength(MPa)' % (self.models['tensile'].predict(best_datapoint.formatForInput())[0]))
            print('Results in a predicted %f yield strength(MPa)' % (self.models['yield'].predict(best_datapoint.formatForInput())[0]))

    def run(self):
        best_loss = None
        best_datapoint = self.generateDatapoint()
        for i in range(self.max_steps):
            loss, datapoint = self.calculateStep(best_datapoint, i, 'all')
            if best_loss is None or loss < best_loss:
                best_datapoint = datapoint
                best_loss = loss
                print('[Step %d] Best %s Loss = %f.' % (i, self.loss_type, best_loss))
        for i in range(self.finetune_max_rounds):
            pre_tune_loss = best_loss
            for key in [*self.categorical_inputs.keys(), *self.range_based_inputs.keys()]:
                loss, datapoint = self.calculateStep(best_datapoint, i, key)
                if loss < best_loss:
                    best_datapoint = datapoint
                    best_loss = loss
            if best_loss < pre_tune_loss:
                print('[Finetune] Best %s Loss = %f.' % (self.loss_type, best_loss))
            else:
                break
        print('==========Scan Finished==========')
        self.printResults(best_datapoint)


    def calculateStep(self, best_datapoint, step_number, target_var):
        if target_var == 'all':
            batch_size = self.step_batch_size
        else:
            batch_size = self.finetune_batch_size
        loss = [0] * batch_size
        datapoints = []
        std = self.step_final_std * (self.max_steps / float(step_number + 1))
        for i in range(batch_size):
            datapoints.append(deepcopy(best_datapoint))
            for key in self.categorical_inputs.keys():
                if target_var == key or target_var == 'all':
                    datapoints[i].categorical_inputs[key] = np.random.choice(self.categorical_inputs[key])
            for key in self.range_based_inputs.keys():
                if target_var == key or target_var == 'all':
                    if max(self.range_based_inputs[key]) != min(self.range_based_inputs[key]):
                        a = (min(self.range_based_inputs[key])-np.mean(best_datapoint.range_based_inputs[key]))/std
                        b = (max(self.range_based_inputs[key])-np.mean(best_datapoint.range_based_inputs[key]))/std
                        datapoints[i].range_based_inputs[key] = round(
                         float(truncnorm.rvs(a, b, loc=np.mean(best_datapoint.range_based_inputs[key]), scale=std)), 2)
                    else:
                        datapoints[i].range_based_inputs[key] = min(self.range_based_inputs[key])
            loss[i] = self.calculateLoss(datapoints[i])
        return min(loss), datapoints[loss.index(min(loss))]