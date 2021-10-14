import timeit

def memoizedCollatz(n, nums):
    cache = dict()

    if n in cache:
        print(cache[n])
        return True
    else:
        cache[n] = nums
        return False

def collatz():
    n = int(input("Please enter a positive integer: "))
    numOriginal = n
    nums = []
    if memoizedCollatz(n, nums):
        memoizedCollatz(n, nums)
    while n != 1:
        nums.append(n)
        if n % 2 == 0:
            n = n/2
        elif n % 2 == 1 and n > 1:
            n = 3*n + 1
    nums.append(n)
    print(nums)
    memoizedCollatz(numOriginal, nums)


collatz()

