class MarkEveryOtherClassifier:
    def __init__(self):
        self.last = 1

    def fit(self, *args):
        pass

    def predict_one(self):
        pred = 1 - self.last
        self.last = pred
        return pred

    def predict(self, xs):
        return [self.predict_one() for _ in xs]
