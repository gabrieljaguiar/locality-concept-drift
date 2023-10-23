from .evaluator import Evaluator
import math
from typing import Dict

# TO DO ADD PMAUC


class MultiClassEvaluator(Evaluator):
    def __init__(self, windowSize, numberOfClasses=2):
        super().__init__(windowSize, numberOfClasses)

    def addResult(self, instance: Dict[float, int], probabilties: Dict):
        _, y = instance
        classVotes = [probabilties.get(i, 0) for i in range(self.numberOfClasses)]
        pred_index = classVotes.index(max(classVotes))
        y_index = [i for i in range(self.numberOfClasses)].index(y)

        #prediction = sorted(list(probabilties.keys()))[pred_index]
        #prediction = probabilties.get(pred_index, 0)
        prediction = pred_index
        
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

            self.columnKappa[class_to_be_removed] -= 1
            self.rowKappa[
                self.prediction_value[self.totalObservedInstances % self.windowSize]
            ] -= 1

            self.correctPositivePredictions -= (
                self.predictions[self.totalObservedInstances % self.windowSize]
                if self.window[self.totalObservedInstances % self.windowSize] == 1
                else 0
            )

        self.predictions[self.totalObservedInstances % self.windowSize] = (
            1 if prediction == y else 0
        )
        self.prediction_value[
            self.totalObservedInstances % self.windowSize
        ] = pred_index
        self.window[self.totalObservedInstances % self.windowSize] = y_index

        self.rowKappa[pred_index] += 1
        self.columnKappa[y_index] += 1
        self.cm[y_index][pred_index] += 1
        self.totalObservedInstances += 1

    def getClassRecall(self, classIdx):
        tp = self.cm[classIdx][classIdx]
        fn = sum(self.cm[:][classIdx]) - self.cm[classIdx][classIdx]
        if tp + fn:
            recall = tp / (tp + fn)
        else:
            recall = 0
        return recall

    def getGMean(self):
        gmean = 1
        for i in range(0, self.numberOfClasses):
            gmean *= self.getClassRecall(classIdx=i)
        try: 
            return math.pow(gmean, 1 / self.numberOfClasses)
        except:
            print (gmean)
            print (1/self.numberOfClasses)

    def getKappa(self):
        p0 = self.getAccuracy()
        pc = 0.0

        pc = sum(
            [
                (self.rowKappa[i] / self.windowSize)
                * (self.columnKappa[i] / self.windowSize)
                for i in range(0, self.numberOfClasses)
            ]
        )
        
        if pc == 1:
            return 0

        return (p0 - pc) / (1.0 - pc)
