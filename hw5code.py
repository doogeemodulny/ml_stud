import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    if feature_vector.size == 1:
        return np.array([]), np.array([]), None, None

    sorted_feature_vector = np.sort(feature_vector)
    thresholds = 0.5 * (sorted_feature_vector[:-1] + sorted_feature_vector[1:])
    thresholds = thresholds[thresholds > sorted_feature_vector[0]]
    thresholds = thresholds[thresholds < sorted_feature_vector[-1]]
    thresholds2 = np.tile(thresholds, (feature_vector.size, 1)).T
    feature_vectors = np.tile(feature_vector, (thresholds.size, 1))
    target_vectors = np.tile(target_vector, (thresholds.size, 1))

    left_subtrees = feature_vectors < thresholds2
    right_subtrees = feature_vectors > thresholds2

    left_nodes = np.where(left_subtrees, feature_vectors, np.nan)
    right_nodes = np.where(right_subtrees, feature_vectors, np.nan)

    def gini(left, right, target):
        p0_left = np.where((~np.isnan(left)), target, np.nan)
        p0_right = np.where((~np.isnan(right)), target, np.nan)

        def count_zeros_ones(row):
            counts = np.bincount(row[~np.isnan(row)].astype(int))
            num_zeros = counts[0] if len(counts) > 1 else 0
            num_ones = counts[1] if len(counts) > 1 else 0
            return (num_zeros, num_ones)

        counts_left = np.apply_along_axis(count_zeros_ones, axis=1, arr=p0_left)
        counts_right = np.apply_along_axis(count_zeros_ones, axis=1, arr=p0_right)

        r_l = np.sum(counts_left, axis=1)
        r_r = np.sum(counts_right, axis=1)
        r = r_l + r_r

        nonzero_indices = np.where(r != 0)[0]  # Находим индексы, в которых r не равно 0

        r_l = r_l[nonzero_indices][:, np.newaxis]
        r_r = r_r[nonzero_indices][:, np.newaxis]
        r = r[nonzero_indices][:, np.newaxis]

        counts_left = counts_left[nonzero_indices]
        counts_right = counts_right[nonzero_indices]
        p0_left_probas = counts_left / r_l
        p1_left_probas = 1 - p0_left_probas
        p0_right_probas = counts_right / r_r
        p1_right_probas = 1 - p0_right_probas

        h_l = np.ones(p0_left_probas.shape) - p0_left_probas ** 2 - p1_left_probas ** 2
        h_r = np.ones(p0_right_probas.shape) - p0_right_probas ** 2 - p1_right_probas ** 2
        ginis = -r_l / r * h_l - r_r / r * h_r
        return ginis

    ginis = gini(left_nodes, right_nodes, target_vectors)
    if ginis.size > 0:
        ginis = ginis[:, 0]
        gini_best = np.max(ginis)
        threshold_best = thresholds[np.argmax(ginis)]
        return thresholds, ginis, threshold_best, gini_best
    return np.array([]), np.array([]), None, None


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=1, min_samples_leaf=1):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, current_depth):
        if np.all(sub_y != sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(0, sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}
            if feature_type == "real":
                feature_vector = np.array(sub_X[:, feature], dtype=np.float64)
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                        ratio[key] = current_count / current_click
                    else:
                        ratio[key] = 0
                sorted_categories = list(map(lambda x: x[1], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(map(lambda x: x[0], zip(sorted(ratio.items(), key=lambda x: x[1]),
                                                              list(range(len(sorted_categories))))))
                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if len(np.unique(feature_vector)) == 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)

            if threshold is None or sub_y.size < self._min_samples_leaf:
                continue

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold
                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None or current_depth == 0 or sub_y.size <= self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
            node["feature_type"] = "real"
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
            node["feature_type"] = "categorical"
        else:
            raise ValueError

        current_depth -= 1
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], current_depth)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], current_depth)

    def _predict_node(self, x, node):
        # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
        if node["type"] == "terminal":
            return node["class"]
        if node["feature_type"] == "real":
            if float(x[node["feature_split"]]) < float(node["threshold"]):
                return self._predict_node(x, node["left_child"])
            if float(x[node["feature_split"]]) >= float(node["threshold"]):
                return self._predict_node(x, node["right_child"])
        if node["feature_type"] == "categorical":
            if x[node["feature_split"]] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            if x[node["feature_split"]] not in node["categories_split"]:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        if not self._max_depth:
            self._max_depth = y.size * X.shape[1]

        self._min_samples_split = self._min_samples_split if type(self._min_samples_split) is int else np.ceil(
            self._min_samples_split * y.size)
        self._min_samples_leaf = self._min_samples_leaf if type(self._min_samples_leaf) is int else np.ceil(
            self._min_samples_leaf * y.size)

        self._fit_node(X, y, self._tree, self._max_depth)
        return self

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

