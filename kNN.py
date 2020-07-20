import numpy as np


class kNN():
    def __init__(self, data, class_tags, numb_classes):
        """
        :param data: n Training Vectors in R^m
        :param class_tags: n Class tags for the training vectors
        :param numb_classes: total number of classes
        """
        self.T = data
        self.T_class = class_tags
        self.numb_T = self.T.shape[1]  # safe number of training-vectors
        self.numb_classes = numb_classes

    def classify(self, vector, kNN):
        """
        :param vector: R^m Vector to classify
        :param kNN: number of next neighbours to look at
        :return: returns the class the vector belongs to
        """
        # Vector to safe all distances
        distance = np.zeros((1, self.numb_T))

        # compute distance between all vectors and the vector to classify
        for x in range(0, self.numb_T):
            # euclidean distance
            distance[0, x] = np.linalg.norm(np.subtract(vector, self.T[:, x].reshape(self.T.shape[0], 1)), 2)

        # sorting distance from small to big
        # and saving the indices
        sorted_dist_index = np.argsort(distance)
        # take the first k indices from sorted index array
        kNN_Vec = sorted_dist_index[0, :kNN]

        # Count the number of times
        # a certain class is in the k next neighbours
        # of our vector to classify
        class_counter = np.zeros((1, self.numb_classes))
        for x in range(0, kNN):
            class_index = self.T_class[0, kNN_Vec[x]]
            class_counter[0, class_index] += 1

        # final class is most often next
        # to the vector to classify
        final_class = np.argmax(class_counter)
        return final_class
