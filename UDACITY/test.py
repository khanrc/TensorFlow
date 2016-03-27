# softmax

scores = [3.0, 1.0, 0.2]

import numpy as np

def softmax(x):
    se = np.exp(x)
    sums = np.sum(se, axis=0)
    return se/sums

scores = np.array(scores)
print(scores)
print(scores*10)
print(softmax(scores))
print(softmax(scores*10))

exit(1)

# plot softmax curve
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

# print(x)
# scores = [[2.0, 1.0, 0.1],[2.0, 1.0, 0.1],[2.0, 1.0, 0.1],[2.0, 1.0, 0.1]]
print(scores)
print(np.sum(scores, axis=0))
print(softmax(scores))
print(np.sum(softmax(scores), axis=0))
# print(np.sum(softmax(scores), axis=1))

# se = np.exp(scores)
# s = np.sum(np.exp(scores), axis=1)
# print(se)
# print(s)
# print((se.T/s).T)
# print(np.divide(se.T, s).T)

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()