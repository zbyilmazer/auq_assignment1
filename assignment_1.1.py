import numpy as np

def mean_impl(arr=None):
    if arr is None:
        return None
    sum = 0
    for i in arr:
        sum += i
    return sum/len(arr)

def var_impl(arr=None):
    if arr is None:
        return None
    mean = mean_impl(arr)
    sum = 0
    for i in arr:
        sum += (i - mean)**2
    return sum/(len(arr)-1)

if __name__ == "__main__":
    arr = np.array([1.3, 1.7, 1.0, 2.0, 1.3, 1.7, 2.0, 2.3, 2.0, 1.7, 1.3, 1.0, 2.0, 1.7, 1.7, 1.3, 2.0])
    mean_np = np.mean(arr)
    var_np = np.var(arr)
    mean = mean_impl(arr.tolist())
    var = var_impl(arr.tolist())
    print("Mean (numpy):", mean_np)
    print("Mean (implementation):", mean)
    print("Variance (numpy):", var_np)
    print("Variance (implementation):", var)