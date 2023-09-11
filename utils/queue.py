class Queue:
    def __init__(self, size: int) -> None:
        self.queue = [0] * size
        self.queueSize = 0
        self.size = size

    def insert(self, element: object):
        self.queue.pop(0)
        self.queue.append(element)
        if self.queueSize < self.size:
            self.queueSize += 1

    def getQueue(self):
        return self.queue

    def getNumberOfElements(self):
        return self.queueSize

    def setQueue(self, listQ):
        self.queue = listQ
        self.size = len(listQ)
        self.queueSize = self.size
