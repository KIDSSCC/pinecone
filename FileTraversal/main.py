import os
import zipfile


def unzip_data(src_path, target_path):
    """
    将一个压缩包解压到指定的目录
    :param src_path:压缩包路径
    :param target_path:目标路径
    :return:
    """
    if not os.path.isdir(target_path):
        z = zipfile.ZipFile(src_path, 'r')
        z.extractall(path=target_path)
        z.close()


size_dict = {}
type_dict = {}


def get_size_type(path):
    """
    统计一个目录下所有类型文件的数量以及大小
    :param path:源路径
    :return:
    """
    files = os.listdir(path)
    for filename in files:
        temp_path = os.path.join(path, filename)
        if os.path.isdir(temp_path):
            # 对于文件夹，递归处理
            get_size_type(temp_path)
        elif os.path.isfile(temp_path):
            # 对于文件，统计其类型和大小
            type_name = os.path.splitext(temp_path)[1]
            if not type_name:
                # 没有后缀名，记为None
                type_dict.setdefault("None", 0)
                type_dict["None"] += 1
                size_dict.setdefault("None", 0)
                size_dict["None"] += os.path.getsize(temp_path)
            else:
                # 有后缀名，按照自己的后缀名保存
                type_dict.setdefault(type_name, 0)
                type_dict[type_name] += 1
                size_dict.setdefault(type_name, 0)
                size_dict[type_name] += os.path.getsize(temp_path)


if __name__ == '__main__':
    print('Waiting for calculate')
    # 需要统计的路径
    path = "E:\\2023年春"
    get_size_type(path)
    for each_type in type_dict.keys():
        print("%5s下共有【%5s】的文件【%5d】个,占用内存【%7.2f】MB" %
              (path, each_type, type_dict[each_type], size_dict[each_type]/(1024*1024)))
    print("总文件数:  【%d】" % (sum(type_dict.values())))
    print("总内存大小:【%.2f】GB" % (sum(size_dict.values()) / (1024 ** 3)))
