import random
import matplotlib.pyplot as plt


def getRandomPoint():
    return random.uniform(-1, 1), random.uniform(-1, 1)


def evaluate(function, point):
    return 1 if (function(point[0]) < point[1]) else -1


def getFunction(targetPoint1, targetPoint2):
    slope = (targetPoint1[1] - targetPoint2[1]) / (targetPoint1[0] - targetPoint2[0])
    y_intercept = targetPoint1[1] - slope * targetPoint1[0]
    return lambda x: x * slope + y_intercept


def dot_product(values, weights):
    return sum(value * weight for value, weight in zip(values, weights))


def get_charactor_vector(vector):
    return vector[0], vector[1], 1


random1 = getRandomPoint()
random2 = getRandomPoint()
f = getFunction(random1, random2)
plt.plot([-1, 1], [f(-1), f(1)], color='b')
w = [0, 0, 0]
training_data = []
N = 100
for i in range(N):
    p = getRandomPoint()
    training_data.append((get_charactor_vector(p), evaluate(f, p)))
    color = 'r' if evaluate(f, p) == 1 else 'c'
    plt.scatter(p[0], p[1], color=color)

iteration = 0
while True:
    disagreed = 0
    for vector, expected in training_data:
        result = 1 if dot_product(vector, w) > 0 else -1
        if expected != result:
            disagreed += 1
            result = evaluate(f, vector)
            for index, value in enumerate(vector):
                iteration += 1
                w[index] += expected * value
    if disagreed == 0:
        break

print iteration

if w[1] == 0:
    pass
else:
    plt.plot([-1, 1], [w[0]/w[1] - w[2]/w[1], -w[0]/w[1] - w[2]/w[1]], color='g')
plt.xlim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)
plt.show()
