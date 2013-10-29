from numpy import *
import numpy
from numpy.linalg import *
import random
import math
import matplotlib.pyplot as plt

def calcSampleNumber(n, ipusilon, delta, growth_func):
    return delta >= 4.0 * growth_func(2 * n) * exp(- 1.0 / 8 * (ipusilon ** 2) * n)


def parrondo(n):
    func = lambda n, ipusilon: sqrt(1.0 / n * (2.0 * ipusilon + log(6.0 * growth_func(2 * n) / delta)))
    for i in arange(0.01, 100, 0.01):
        if func(n, i) < i:
            return i
    return inf


def devroye(n):
    func = lambda n, ipusilon: sqrt(1.0 / (2.0 * n) * (4 * ipusilon * (1 + ipusilon) + 930))
    for i in arange(0.01, 100, 0.01):
        if func(n, i) < i:
            return i
    return inf

# 1.
print "#1"
growth_func = lambda x: x ** 10
print "400000: " + str(calcSampleNumber(400000, 0.05, 0.05, growth_func))
print "420000: " + str(calcSampleNumber(420000, 0.05, 0.05, growth_func))
print "440000: " + str(calcSampleNumber(440000, 0.05, 0.05, growth_func))
print "460000: " + str(calcSampleNumber(460000, 0.05, 0.05, growth_func))
print "480000: " + str(calcSampleNumber(480000, 0.05, 0.05, growth_func))


# 2.
print "#2"
delta = 0.05
growth_func = lambda x: x ** 50
a_func = lambda n: sqrt(8.0 / n * log(4 * growth_func(2.0 * n) / delta))
print "a: " + str(a_func(10000))
b_func = lambda n: sqrt((2.0 * log(2.0 * n * growth_func(n))) / n) + sqrt(2.0 / n + log(1.0 / delta)) + 1.0 / n
print "b: " + str(b_func(10000))
print "c: " + str(parrondo(10000))
print "d: " + str(devroye(10000))

# 3.
print "#3"
print "a: " + str(a_func(5))
print "b: " + str(b_func(5))
print "c: " + str(parrondo(5))
print "d: " + str(devroye(5))

# 4.
print "#4"


def getPoints(func):
    x1 = random.uniform(-1, 1)
    y1 = func(x1)
    x2 = random.uniform(-1, 1)
    y2 = func(x2)
    return (x1, y1), (x2, y2)


def targetFunc(x):
    return math.sin(math.pi * x)


def calc(N, targetFunc, getCoefficientFunc):
    totalA = 0
    totalB = 0
    totalC = 0
    for i in range(0, N):
        p1, p2 = getPoints(targetFunc)
        a, b, c = getCoefficientFunc(p1, p2)
        totalA += a
        totalB += b
        totalC += c
    averageA = totalA / N
    averageB = totalB / N
    averageC = totalC / N
    averageFunc = lambda x: averageA * (x ** 2) + averageB * x + averageC
    print "average func: %f x^2 + %f x + %f" % (averageA, averageB, averageC)
    total_bias = 0
    for i in range(0, N):
        x = random.uniform(-1, 1)
        total_bias += (averageFunc(x) - targetFunc(x)) ** 2
    average_bias = total_bias / N
    print "average bias: %f" % average_bias
    total_variance = 0
    for i in range(0, N):
        p1, p2 = getPoints(targetFunc)
        a, b, c = getCoefficientFunc(p1, p2)
        x = random.uniform(-1, 1)
        value = a * (x ** 2) + b * x + c
        total_variance += (value - averageFunc(x)) ** 2
    average_variance = total_variance / N
    print "average variance: %f" % average_variance
    print "out of sample error: %f" % (average_bias + average_variance)


def getFunc2(p1, p2):
    slope = (p1[0] * p1[1] + p2[0] * p2[1]) / (p1[0] ** 2 + p2[1] ** 2)
    return 0, slope, 0

print "y = ax"
calc(100, targetFunc, getFunc2)

print "y = b"
calc(100, targetFunc, lambda p1, p2: (0, 0, (p1[1] + p2[1]) / 2))

print "y = ax + b"

def _get_approximate_f(x, y, rank):
    matrix = []
    for i in range(rank, 0, -1):
        matrix.append(x ** i)
    matrix.append(ones(len(x)))
    A = vstack(matrix).T
    return lstsq(A, y)[0]

def fit(xs, ys, rank = 1):
    x = numpy.array(xs)
    y = numpy.array(ys)
    return _get_approximate_f(x, y, rank)

def getCoeff3(p1, p2):
    coeffs = fit([p1[0], p2[0]], [p1[1], p2[1]], 1)
    return 0, coeffs[0], coeffs[1]

calc(100, targetFunc, getCoeff3)

print "y = a * x ^ 2"
calc(100, targetFunc, lambda p1, p2: (((p1[0] ** 2) * p1[1] + ((p2[0] ** 2) * p2[1])) / (p1[0] ** 3 + p2[1] ** 3), 0, 0))

