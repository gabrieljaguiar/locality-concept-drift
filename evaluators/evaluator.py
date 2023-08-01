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
        classVotes = [
            probabilties.get(list(probabilties.keys())[i])
            for i in range(self.numberOfClasses)
        ]
        pred_index = classVotes.index(max(classVotes))
        y_index = list(probabilties.keys()).index(y)

        prediction = list(probabilties.keys())[pred_index]

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
        ] = prediction
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


if __name__ == "__main__":
    from river.tree import HoeffdingTreeClassifier
    import concept_drift

    windowsize = 500
    evaluator = Evaluator(windowSize=windowsize)

    stream1 = synth.Agrawal(classification_function=0, seed=42)
    stream2 = synth.Agrawal(classification_function=8, seed=42)

    conceptDriftStream = concept_drift.ConceptDriftStream(
        stream1, stream2, width=1, position=4000, angle=0
    )

    model = HoeffdingTreeClassifier(
        grace_period=100, delta=1e-5, nominal_attributes=["elevel", "car", "zipcode"]
    )

    idx = 0

    for x, y in conceptDriftStream.take(200):
        model.learn_one(x, y)

    for x, y in conceptDriftStream.take(6000):
        idx += 1
        y_hat = model.predict_proba_one(x)
        evaluator.addResult((x, y), y_hat)
        model.learn_one(x, y)
        if idx % windowsize == 0:
            print("Accuracy {}: {}%".format(idx, evaluator.getAccuracy()))
