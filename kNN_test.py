import numpy as np
import matplotlib.pyplot as plt
import kNN

# Producing a grid of equidistant data in [0,1] x [0,1]
Data = np.zeros((2, 100))
Data_tag = np.zeros((1, 100))
for x in range(0, 10):
    for y in range(0, 10):
        Data[:, x * 10 + y] = np.array([[0.1 * x], [0.1 * y]]).reshape(2)
        if Data[0, x * 10 + y] - Data[1, x * 10 + y] <= 0:
            Data_tag[:, x * 10 + y] = 0
        else:
            Data_tag[:, x * 10 + y] = 1

# Plotting trainings-data
for points in range(0, Data.shape[1]):
    if Data_tag[0, points] == 0:
        plt.plot(Data[0, points], Data[1, points], 'bo', label='Class 0')
    else:
        plt.plot(Data[0, points], Data[1, points], 'ro', label='Class 1')

# Initialize the classifier
kNN1 = kNN.kNN(Data, Data_tag.astype(int), 2)

# Initialize the vector to test
test = np.array([[0.55], [0.55]])

# Plot point to classify
plt.plot(test[0, 0], test[1, 0], 'gp', markersize=10, label='Test-Vector')
plt.show()

# Compute class
classified = kNN1.classify(test, 5)

print("Object belongs to class {}".format(classified))
