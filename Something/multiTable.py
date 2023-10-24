def multi_table():
    """
    打印九九乘法表
    :return: str，完整的九九乘法表
    """
    s = ''
    for i in range(1,10):
        for j in range(1,i+1):
            s += '{}*{}={}'.format(i,j,i*j)+' '
        s += '\n'
    return s


if __name__ == '__main__':
    s = multi_table()
    print(s)
