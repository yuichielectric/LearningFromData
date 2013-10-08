# -*- coding: utf-8 -*-

import random
# import matplotlib.pyplot as plt


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

# 実験用のパラメータ
## イテレーション実施数
ITERATION = 100
## トレーニングデータの数
N = 100
## ターゲット関数と値が一致しない確率を求めるテストを実行する回数
EQUALITY_TEST_COUNT = 10000

total = 0
total_not_matched = 0

for i in range(ITERATION):
    random1 = getRandomPoint()
    random2 = getRandomPoint()
    f = getFunction(random1, random2)
    # plt.plot([-1, 1], [f(-1), f(1)], color='b')
    w = [0, 0, 0]
    training_data = []
    for i in range(N):
        p = getRandomPoint()
        training_data.append((get_charactor_vector(p), evaluate(f, p)))
        color = 'r' if evaluate(f, p) == 1 else 'c'
        # plt.scatter(p[0], p[1], color=color)

    iteration = 0
    while True:
        disagreed = []
        for vector, expected in training_data:
            result = 1 if dot_product(vector, w) > 0 else -1
            if expected != result:
                disagreed.append((vector, expected))

        if len(disagreed) == 0:
            break
        else:
            misclassified = disagreed[random.randint(0, len(disagreed) - 1)]
            for index, value in enumerate(misclassified[0]):
                w[index] += misclassified[1] * value
            iteration += 1
        if disagreed == 0:
            break
    total += iteration

    not_matched = 0
    for i in range(EQUALITY_TEST_COUNT):
        p = getRandomPoint()
        expected = evaluate(f, p)
        actual = 1 if dot_product(get_charactor_vector(p), w) > 0 else -1
        if expected != actual:
            not_matched += 1
    total_not_matched += not_matched

print "Average iteration count: %d" % (total / ITERATION)
print "Average P[f(x) != g(x)]: %f" % (float(total_not_matched) / (ITERATION * EQUALITY_TEST_COUNT))

#if w[1] == 0:
#    pass
#else:
#    plt.plot([-1, 1], [w[0]/w[1] - w[2]/w[1], -w[0]/w[1] - w[2]/w[1]], color='g')
#plt.xlim(-1.0, 1.0)
#plt.ylim(-1.0, 1.0)
#plt.show()
