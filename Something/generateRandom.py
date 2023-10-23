import random


def random_int(ranges=[1, 100], num=1):
    """
    在一定范围内，产生指定数量的随机整数
    :param ranges: 随机数范围
    :param num: 随机数数量
    :return: 包含指定随机数的list
    """
    if ranges[0] > ranges[1]:
        print('error,left boundary greater than right boundary')
        return []
    res = []
    for i in range(num):
        res.append(random.randint(ranges[0], ranges[1] + 1))
    return res


def random_float(ranges=[1, 100], num=1):
    """
    在一定范围内，产生指定数量的随机浮点数
    :param ranges: 随机数范围
    :param num: 随机数数量
    :return: 包含指定随机数的list
    """
    if ranges[0] > ranges[1]:
        print('error,left boundary greater than right boundary')
        return []
    res = []
    for i in range(num):
        res.append(random.random()*(ranges[1]-ranges[0]) + ranges[0])
    return res


def quick_sort(arr):
    """
    快速排序算法
    :param arr:待排序的数组
    :return:排序结果
    """
    if len(arr) < 2:
        # 递归中止条件
        return arr
    mid_i = len(arr) // 2
    left, mid, right = [], [], []
    for num in arr:
        # 根据和中间元素的比较分为了三个新的list
        if num < arr[mid_i]:
            left.append(num)
        elif num == arr[mid_i]:
            mid.append(num)
        elif num > arr[mid_i]:
            right.append(num)
    return quick_sort(left)+mid+quick_sort(right)


if __name__ == '__main__':
    origin = random_int([1, 100], 15)
    print(origin)
    after_sort = quick_sort(origin)
    print(after_sort)

