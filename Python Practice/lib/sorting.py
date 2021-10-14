import time
import numpy as np

time_min = time.time()

def bubbleSort(arr):
    for i in range(len(arr)):
        for j in range(0, len(arr)-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    print(arr, time_min)

arr = np.random.rand(2000, 1)

bubbleSort(arr)

print(sorted(arr), time_min)
