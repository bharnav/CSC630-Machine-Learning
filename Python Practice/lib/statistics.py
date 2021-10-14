from collections import Counter
import math

def mean(arr):
    tot = 0
    count = 0
    for i in range(len(arr)-1):
        tot += arr[i]
        count = count + 1
    return float(tot)/count

def median(arr):
    n = len(arr)
    arr = sorted(arr)
    if n%2 == 0:
        return arr[n/2]
    else:
        return (float(arr[(n-1)/2] + float(arr[(n+1)/2]))/2.0)

def mode(arr):
    nums = dict(Counter(arr))
    sort = (sorted(list(nums.values())))
    return sort[len(sort) - 1]

def standardDeviation(arr):
    totM = 0
    count = 0
    for i in range(len(arr)-1):
        totM += arr[i]
        count = count + 1
    mean=float(totM)/count
    tot = 0
    for i in range(len(arr) - 1):
        tot += (arr[i] - mean)**2
    return math.sqrt(tot)

a = [11, 16, 14, 19, 1, 13, 15, 15, 2, 6, 4, 20, 17, 8, 18, 22, 25, 11, 18, -7]
b = [17, 15, 7, 12, 5, 20, 18, 22, 11, 2, 9, 0, 10, 11, 6, 17, 9, 10, 6]
c = [6, 16, 1, 6, 14, 5, 5, 15, 6, 11, 8, 15, 10, 3, 15, 10, 5, 14, 17, 13]

mean = mean(b)
median = median(b)
mode = mode(b)

d = a + b + c
print(d)

print(standardDeviation(d))
