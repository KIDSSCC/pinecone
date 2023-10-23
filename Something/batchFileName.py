import os


def change_file_name(dir_path):
    """
    批量的更改一个路劲下所有文件的文件名，
    :param dir_path:目标目录
    :return:
    """
    files = os.listdir(dir_path)
    for f in files:
        oldname = os.path.join(dir_path, f)
        newname = os.path.join(dir_path, 'rename_' + f)
        os.rename(oldname, newname)
        print(oldname, '======>', newname)


def delete_prefix(dir_path, prefix):
    """
    批量的删除一个路径下文件名包含某个前缀的文件的文件名中的前缀
    :param dir_path:目标目录
    :param prefix:要删除的前缀
    :return:
    """
    files = os.listdir(dir_path)
    for f in files:
        oldname = os.path.join(dir_path, f)
        newname = os.path.join(dir_path, f[len(prefix):])
        os.rename(oldname, newname)
        print(oldname, '======>', newname)


if __name__ == '__main__':
    dir_path = 'C:\\Users\\15334\\Desktop\\征集群众意见'
    delete_prefix(dir_path, 'rename_')