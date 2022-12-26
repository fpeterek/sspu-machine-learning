
class Leaf:
    def __init__(self, label):
        self.label = label

    def predict(self, x):
        return self.label

    def __call__(self, x):
        return self.predict(x)


class Tree:
    def __init__(self, attr, threshold, left, right):
        self.attr = attr
        self.threshold = threshold
        self.left = left
        self.right = right

    def predict(self, x):
        if x[self.attr] <= self.threshold:
            return self.left(x)
        return self.right(x)

    def __call__(self, x):
        return self.predict(x)


def gini_idx(xs, attr, threshold):
    pass


def find_min_max(xs, attr):
    min_x = max_x = xs[0][attr]

    for x in xs:
        min_x = min(min_x, x[attr])
        max_x = max(max_x, x[attr])

    return min_x, max_x


def split_attr(xs, attr) -> tuple[float, float]:
    min_x, max_x = find_min_max(xs, attr)

    if min_x == max_x:
        return 0, min_x

    step = (max_x - min_x) / 10
    threshold = min_x
    best_gini = 0
    best_threshold = threshold

    while threshold <= max_x:

        gini = gini_idx(xs, attr, threshold)

        if gini > best_gini:
            best_gini = gini
            best_threshold = threshold

        threshold += step

    return best_gini, best_threshold


def find_optimal_split(xs):
    best_attr, best_threshold, best_gini = 0, 0, 0

    for attr in range(len(xs[0])):
        gini, threshold = split_attr(xs, attr)
        if gini > best_gini:
            best_attr = attr
            best_threshold = threshold
            best_gini = gini

    return best_attr, best_threshold
