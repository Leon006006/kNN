import numpy as np
import matplotlib.pyplot as plt
import kNN


Data = np.zeros((2, 100))
Data_tag = np.zeros((1, 100))
for x in range(0, 10):
    for y in range(0, 10):
        Data[:, x * 10 + y] = np.array([[0.1 * x], [0.1 * y]]).reshape(2)
        if Data[0, x * 10 + y] - Data[1, x * 10 + y] <= 0:
            Data_tag[:, x * 10 + y] = 0
        else:
            Data_tag[:, x * 10 + y] = 1
"""
print(Data_tag)
plt.plot(Data[0, :], Data[1, :])
plt.show()
"""

kNN1 = kNN.kNN(Data, Data_tag.astype(int), 2)
test = np.array([[0.6], [0.5]])
kNN1.classify(test, 5)