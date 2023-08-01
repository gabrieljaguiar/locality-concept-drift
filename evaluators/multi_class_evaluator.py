from .evaluator import Evaluator
import math


def printTable(table):
    for i in range(0, len(table[0])):
        print("{}".format(table[i]))


class MultiClassEvaluator(Evaluator):
    def __init__(self, windowSize, numberOfClasses=2):
        super().__init__(windowSize, numberOfClasses)

    def getClassRecall(self, classIdx):
        tp = self.cm[classIdx][classIdx]
        fn = sum(self.cm[classIdx][:])
        return tp / (tp + fn)

    def getGMean(self):
        gmean = 1
        for i in range(0, self.numberOfClasses):
            gmean *= self.getClassRecall(classIdx=i)

        return math.pow(gmean, 1 / self.numberOfClasses)
