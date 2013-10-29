# -*- coding: utf-8 -*-

import random


N = 1000
ITERATION = 1000
OUT_OF_SAMPLE_COUNT = 1000
TARGET_FUNCTION = lambda x: sign(x[0] ** 2 + x[1] ** 2 - 0.6)


def getRandomPoint():
    return random.uniform(-1, 1), random.uniform(-1, 1)


def sign(x):
    return 1 if x > 0 else -1


def flipSign(x):
    return 1 if x == -1 else -1


def getTrainingData(n):
    data = []
    for i in range(n):
        p = getRandomPoint()
        value = TARGET_FUNCTION(p)
        if random.randint(0, 9) == 0:
            value = flipSign(value)
        data.append((transformVector(p), value))
    return data


def getFeatureVector(p):
    return 1, p[0], p[1]


def getLinearRegressionWeight(x, y):
    from numpy import dot, transpose
    from numpy.linalg import inv
    return dot(dot(inv(dot(transpose(x), x)), transpose(x)), y)


def transformVector(x):
    return 1, x[0], x[1], x[0] * x[1], x[0] ** 2, x[1] ** 2

in_sample_error_count = 0
out_of_sample_error_count = 0

for i in range(ITERATION):
    if i % 50 == 0:
        print "%d iteration finished." % i
    w = [0, 0, 0]
    x = []
    y = []
    samples = getTrainingData(N)
    for point, expected in samples:
        x.append(point)
        y.append([expected])

    w = getLinearRegressionWeight(x, y)

    for point, expected in samples:
        from numpy import dot, transpose
        if expected != sign(dot(transpose(w), point)):
            in_sample_error_count += 1

    for vector, expected in getTrainingData(OUT_OF_SAMPLE_COUNT):
        if expected != sign(dot(transpose(w), vector)):
            out_of_sample_error_count += 1


print "in sample error rate: %f" % (float(in_sample_error_count) / (ITERATION * N))
if OUT_OF_SAMPLE_COUNT > 0:
    print "out of sample error rate: %f" % (float(out_of_sample_error_count) / (ITERATION * OUT_OF_SAMPLE_COUNT))
