import re


def calculate(n1, n2, operator):
    """
    将两个操作数按照指定的运算符进行处理
    :param n1:float
    :param n2:float
    :param operator:操作符，+-/*
    :return:运算结果
    """
    result = 0
    if operator == '+':
        result = n1 + n2
    if operator == '-':
        result = n1 - n2
    if operator == '*':
        result = n1 * n2
    if operator == '/':
        result = n1 / n2
    return result


def is_operator(e):
    opers = ['+', '-', '*', '/', '(', ')']
    return True if e in opers else False


def formula_format(formula):
    formula = re.sub(' ', '', formula)
    formula_list = [i for i in re.split('(\-\d+\.?\d*)', formula) if i]
