
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


def gini_idx(xs, ys, attr, threshold):
    left_counts = dict()
    left_len = 0
    right_counts = dict()
    right_len = 0

    for idx, x in enumerate(xs):
        if x[attr] <= threshold:
            left_counts[ys[idx]] = left_counts.get(ys[idx], 0) + 1
            left_len += 1
        else:
            right_counts[ys[idx]] = right_counts.get(ys[idx], 0) + 1
            right_len += 1

    left_probs = 0
    right_probs = 0

    for count in left_counts.values():
        left_probs += (count / left_len) ** 2

    for count in right_counts.values():
        right_probs += (count / right_len) ** 2

    left = 1 - left_probs
    right = 1 - right_probs

    return left * (left_len / len(xs)) + right * (right_len / len(xs))


def find_min_max(xs, attr):
    min_x = max_x = xs[0][attr]

    for x in xs:
        min_x = min(min_x, x[attr])
        max_x = max(max_x, x[attr])

    return min_x, max_x


def split_attr(xs, ys, attr) -> tuple[float, float]:
    min_x, max_x = find_min_max(xs, attr)

    if min_x == max_x:
        return 0, min_x

    step = (max_x - min_x) / 10
    threshold = min_x
    best_gini = 1
    best_threshold = threshold

    while threshold <= max_x:

        gini = gini_idx(xs, ys, attr, threshold)

        if gini < best_gini:
            best_gini = gini
            best_threshold = threshold

        threshold += step

    return best_gini, best_threshold


def find_optimal_split(xs, ys, attrs):
    best_attr, best_threshold, best_gini = 0, 0, 1

    for attr in attrs:
        gini, threshold = split_attr(xs, ys, attr)
        if gini < best_gini:
            best_attr = attr
            best_threshold = threshold
            best_gini = gini

    return best_attr, best_threshold


def mode(ys):
    counts = dict()
    for y in ys:
        counts[y] = counts.get(y, 0) + 1

    return max(counts, key=counts.get)


def split_data(xs, ys, attr, threshold):
    lx, ly, rx, ry = [], [], [], []

    for x, y in zip(xs, ys):
        x_coll, y_coll = lx, ly
        if x[attr] > threshold:
            x_coll, y_coll = rx, ry
        x_coll.append(x)
        y_coll.append(y)

    return lx, ly, rx, ry


def create_tree(xs, ys, depth=None, attrs=None):
    if depth is None or depth > len(xs[0]):
        depth = len(xs[0])
    if attrs is None:
        attrs = set(range(len(xs[0])))

    if depth < 1:
        raise ValueError('Tree depth must be 1 or greater')

    if depth == 1:
        return Leaf(mode(ys))

    attr, threshold = find_optimal_split(xs, ys, attrs)

    print(attr, threshold)

    lx, ly, rx, ry = split_data(xs, ys, attr, threshold)

    if not ly:
        return Leaf(mode(ry))
    if not ry:
        return Leaf(mode(ly))

    attrs.remove(attr)
    left = create_tree(lx, ly, depth-1, attrs)
    right = create_tree(rx, ry, depth-1, attrs)
    attrs.add(attr)

    return Tree(attr, threshold, left, right)
