# -*- coding: utf-8 -*-

import random
from numpy import *
from numpy.linalg import *
from numpy.matrixlib import *
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


def get_linear_regression_weight(x, y):
    return dot(dot(inv(dot(transpose(x), x)), transpose(x)), y)


def sign(x):
    return 1 if x > 0 else -1


# 実験用のパラメータ
## イテレーション実施数
ITERATION = 1000
## トレーニングデータの数
N = 10
## ターゲット関数と値が一致しない確率を求めるテストを実行する回数
EQUALITY_TEST_COUNT = 10000
## グラフを表示するかどうかのフラグ
SHOW_GRAPH = False
## wがもとまった後にサンプル外の点の成否を確認する際の点の数
OUT_OF_SAMPLE_COUNT = -1

in_sample_error_count = 0
out_of_sample_error_count = 0
total_iteration = 0

for i in range(ITERATION):
    if i % 50 == 0:
        print "%d iteration finished." % i
    random1 = getRandomPoint()
    random2 = getRandomPoint()
    f = getFunction(random1, random2)
    if SHOW_GRAPH:
        plt.plot([-1, 1], [f(-1), f(1)], color='b')
    w = [0, 0, 0]
    training_data = []
    x = []
    y = []
    samples = []
    for i in range(N):
        p = getRandomPoint()
        samples.append(p)
        x.append(get_charactor_vector(p))
        y.append([evaluate(f, p)])
        if SHOW_GRAPH:
            color = 'r' if evaluate(f, p) == 1 else 'c'
            plt.scatter(p[0], p[1], color=color)

    w = get_linear_regression_weight(x, y)

    for point in samples:
        if evaluate(f, point) != sign(dot(transpose(w), get_charactor_vector(point))):
            in_sample_error_count += 1

    if OUT_OF_SAMPLE_COUNT > 0:
        for i in range(OUT_OF_SAMPLE_COUNT):
            p = getRandomPoint()
            if evaluate(f, p) != sign(dot(transpose(w), get_charactor_vector(p))):
                out_of_sample_error_count += 1

    # Exec perceptron based on linear regression weight.
    iteration = 1
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
    total_iteration += iteration


if SHOW_GRAPH:
    if w[1] == 0:
        pass
    else:
        plt.plot([-1, 1], [w[0]/w[1] - w[2]/w[1], -w[0]/w[1] - w[2]/w[1]], color='g')
    plt.xlim(-1.0, 1.0)
    plt.ylim(-1.0, 1.0)
    plt.show()

print "in sample error rate: %f" % (float(in_sample_error_count) / (ITERATION * N))
if OUT_OF_SAMPLE_COUNT > 0:
    print "out of sample error rate: %f" % (float(out_of_sample_error_count) / (ITERATION * OUT_OF_SAMPLE_COUNT))
print "perceptron exec counts: %f" % (float(total_iteration) / ITERATION)