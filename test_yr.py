import numpy as np


def fun1(c):
    d = np.array([1, 2, 3, 4])
    c = d


def fun2(c):
    d = [1, 2, 3, 4]
    c = d


if __name__ == '__main__':

    a = np.array([1, 2, 3])
    print(f"before a is: {a}")
    fun1(a)
    print(f"after a is: {a}")

    b = [1, 2, 3]
    print(f"before b is: {b}")
    fun2(b)
    print(f"after b is: {b}")