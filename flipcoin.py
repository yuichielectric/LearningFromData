import random

COIN_COUNT = 1000
FLIP_COUNT = 10
EXECUTE_COUNT = 10000

def flip():
    """
     Returns true if the coin is head, false otherwise.
    """
    return random.random() >= 0.5


def countTrue(l):
    return sum(1 for elem in l if elem)


total_1st = 0
total_rand = 0
total_min = 0

for i in range(EXECUTE_COUNT):
    coin_results = []
    min = 10
    for j in range(COIN_COUNT):
        result = []
        count = 0
        for k in range(FLIP_COUNT):
            value = flip()
            result.append(value)
            count += 1 if value else 0
        if min > count:
            min = count
        coin_results.append(result)
    total_1st += countTrue(coin_results[0])
    total_rand += countTrue(coin_results[random.randint(0, COIN_COUNT - 1)])
    total_min += min

    if i % 1000 == 0:
        print "%d exec finished." % i

print float(total_1st) / (EXECUTE_COUNT * FLIP_COUNT)
print float(total_rand) / (EXECUTE_COUNT * FLIP_COUNT)
print float(total_min) / (EXECUTE_COUNT * FLIP_COUNT)