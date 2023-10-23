import os


def change_file_name(dir_path):
    files = os.listdir(dir_path)
    for f in files:
        oldname = os.path.join(dir_path, f)
        newname = os.path.join(dir_path, 'rename_' + f)
        os.rename(oldname, newname)
        print(oldname, '======>', newname)


def delete_prefix(dir_path, prefix):
    files = os.listdir(dir_path)
    for f in files:
        oldname = os.path.join(dir_path, f)
        newname = os.path.join(dir_path, f[len(prefix):])
        os.rename(oldname, newname)
        print(oldname, '======>', newname)


if __name__ == '__main__':
    dir_path = 'C:\\Users\\15334\\Desktop\\征集群众意见'
    delete_prefix(dir_path, 'rename_')