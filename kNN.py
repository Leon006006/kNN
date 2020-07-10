import numpy as np


class kNN():
    def __init__(self, data, class_tags, numb_classes):
        self.T = data
        self.T_class = class_tags
        self.numb_classes = numb_classes

    def classify(self, vector, kNN):
        numb_T = self.T.shape[1]
        distance = np.zeros((1, numb_T))

        for x in range(0, numb_T):
            distance[0, x] = np.linalg.norm(np.subtract(vector, self.T[:, x].reshape(self.T.shape[0], 1)), 2)

        sorted_dist_index = np.argsort(distance)
        kNN_Vec = sorted_dist_index[0, :kNN]
        class_counter = np.zeros((1, self.numb_classes))

        for x in range(0, kNN):
            class_index = self.T_class[0, kNN_Vec[x]]
            class_counter[0, class_index] += 1

        final_class = np.argmax(class_counter)
        print("Objekt gehört zu Klasse {}".format(final_class))