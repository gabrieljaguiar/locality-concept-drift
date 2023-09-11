import random
import math
import river
from river.datasets import synth
from typing import Dict
import numpy as np


class Evaluator:
    def __init__(self, windowSize, numberOfClasses=2):
        self.windowSize = windowSize
        self.totalObservedInstances = 0
        self.predictions = [0] * self.windowSize
        self.prediction_value = [0] * self.windowSize
        self.correctPositivePredictions = 0
        self.numPos = 0
        self.numNeg = 0
        self.window = [0] * self.windowSize
        self.numberOfClasses = numberOfClasses
        self.rowKappa = [0.0] * self.numberOfClasses
        self.columnKappa = [0.0] * self.numberOfClasses
        self.cm = np.zeros((self.numberOfClasses, self.numberOfClasses))

    def addResult(self, instance: Dict[float, int], probabilties: Dict):
        _, y = instance
        # print(probabilties)
        classVotes = [probabilties.get(i, None) for i in range(self.numberOfClasses)]
        pred_index = classVotes.index(max(classVotes))
        y_index = [i for i in range(self.numberOfClasses)].index(y)

        prediction = sorted(list(probabilties.keys()))[pred_index]

        if self.totalObservedInstances > self.windowSize:
            class_to_be_removed = self.window[
                self.totalObservedInstances % self.windowSize
            ]
            if self.predictions[self.totalObservedInstances % self.windowSize] == 1:
                self.cm[class_to_be_removed][class_to_be_removed] -= 1
            else:
                self.cm[class_to_be_removed][
                    self.prediction_value[self.totalObservedInstances % self.windowSize]
                ] -= 1

            self.correctPositivePredictions -= (
                self.predictions[self.totalObservedInstances % self.windowSize]
                if self.window[self.totalObservedInstances % self.windowSize] == 1
                else 0
            )

            if self.window[self.totalObservedInstances % self.windowSize] == 1:
                self.numPos -= 1
            else:
                self.numNeg -= 1

        self.predictions[self.totalObservedInstances % self.windowSize] = (
            1 if prediction == y else 0
        )
        self.prediction_value[
            self.totalObservedInstances % self.windowSize
        ] = pred_index
        self.window[self.totalObservedInstances % self.windowSize] = y_index
        if y_index == 1:
            self.correctPositivePredictions += 1 if prediction == y else 0
            self.numPos += 1
        else:
            self.numNeg += 1
        self.cm[y_index][pred_index] += 1
        self.totalObservedInstances += 1

    def getAccuracy(self) -> float:
        accuracy = (
            (
                sum(self.predictions[: self.totalObservedInstances])
                / self.totalObservedInstances
            )
            if self.totalObservedInstances < self.windowSize
            else sum(self.predictions) / self.windowSize
        )

        return accuracy

    def getGMean(self) -> float:
        try:
            posAccuracy = self.correctPositivePredictions / self.numPos
        except:
            posAccuracy = 0
        try:
            negAccuracy = (
                sum(self.predictions) - self.correctPositivePredictions
            ) / self.numNeg
        except:
            negAccuracy = 0

        if negAccuracy < 0:
            negAccuracy = 0
        if posAccuracy < 0:
            posAccuracy = 0

        return math.sqrt(posAccuracy * negAccuracy)
