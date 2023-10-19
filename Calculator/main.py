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
    """
    判断一个字符是否是一个运算符
    :param e:char,待判断字符
    :return:bool 判断结果
    """
    opers = ['+', '-', '*', '/', '(', ')']
    return True if e in opers else False


def formula_format(formula):
    """
    将原有的字符流转换为token，
    :param formula:str 原表达式
    :return:各个单元组成的list
    """
    formula = re.sub(' ', '', formula)
    # 先粗略的划分为以负号开头的和不以负号开头的
    formula_list = [i for i in re.split('(\-\d+\.?\d*)', formula) if i]
    final_formula = []
    for item in formula_list:
        if len(final_formula) == 0 and re.search('^\-\d+\.?\d*$', item):
            # 如果当前列表为空，说明这是一个负数，而不是减号
            final_formula.append(item)
            continue
        if len(final_formula) > 0:
            if re.search('[\+\-\*\/\(]$', final_formula[-1]):
                # 当前列表不为空，但上一个单元是一个运算符，那么当前单元也要当吃负数来看待
                final_formula.append(item)
                continue
        # 更细粒度的划分，分开操作数与运算符
        item_spilt = [i for i in re.split('([\+\-\*\/\(])', item) if i]
        final_formula += item_spilt
    return final_formula


def decision(tail_op, now_op):
    """
    决定运算符之间的优先顺序
    （左侧的所有运算符优先级低于（，（右侧的运算符除了）意外优先级都高于（
    :param tail_op:char,栈内的上一个运算符
    :param now_op:char,当前新的运算符
    :return:int,标识下一步的操作，-1：压栈操作  0：()的匹配，1：对栈内的运算符进行弹栈计算
    """
    rate1 = ['+', '-']
    rate2 = ['*', '/']
    rate3 = ['(']
    rate4 = [')']
    if tail_op in rate1:
        if now_op in rate2 or now_op in rate3:
            return -1
        else:
            return 1
    elif tail_op in rate2:
        if now_op in rate3:
            return -1
        else:
            return 1
    elif tail_op in rate3:
        if now_op in rate4:
            return 0
        else:
            return -1
    else:
        return -1


def final_calc(formual_list):
    """
    整合之后的计算程序
    :param formual_list:list,原表达式经过处理后的列表
    :return:op_stack：运算符栈，num_stack：操作数栈
    """
    num_stack = []
    op_stack = []
    for e in formual_list:
        operator = is_operator(e)
        if not operator:
            num_stack.append(float(e))
        else:
            while True:
                if len(op_stack) == 0:
                    # 操作符栈为空，必定进行压栈
                    op_stack.append(e)
                    break
                tag = decision(op_stack[-1], e)
                if tag == -1:
                    # 左侧运算符优先级低于右侧，进行压栈处理
                    op_stack.append(e)
                    break
                elif tag == 0:
                    # 左右括号的匹配，右括号不压入栈中，直接弹出一个左括号
                    op_stack.pop()
                    break
                elif tag == 1:
                    # 左侧运算符优先级高于右侧，不断进行弹栈计算
                    op = op_stack.pop()
                    num2 = num_stack.pop()
                    num1 = num_stack.pop()
                    num_stack.append(calculate(num1, num2, op))
    while len(op_stack) != 0:
        op = op_stack.pop()
        num2 = num_stack.pop()
        num1 = num_stack.pop()
        num_stack.append(calculate(num1, num2, op))
    return op_stack, num_stack


if __name__ == '__main__':
    formula = input('请输入表达式：')
    print('算式：', formula)
    formula_list = formula_format(formula)
    _, result = final_calc(formula_list)
    print('计算结果：', result[0])

