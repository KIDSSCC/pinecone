import numpy as np


def gen_configs_recursively_fix(num_res:int, num_apps:int):
    """
    get a resource's allocation space.
    input: resource id, num of groups/apps
    return: a list contains all allocation plans of a resource
    """
    def gen_configs_recursively(u, num_res, a, num_apps):
        if (a == num_apps - 1):
            return None
        else:
            ret = []
            for i in range(1, num_res - u + 1 - num_apps + a + 1):
                confs = gen_configs_recursively(u + i, num_res, a + 1, num_apps)
                if not confs:
                    ret.append([i])
                else:
                    for c in confs:
                        ret.append([i])
                        for j in c:
                            ret[-1].append(j)
            return ret
    res_config = gen_configs_recursively(0, num_res, 0, num_apps)
    print(res_config)
    for i in range(len(res_config)):
        other_source = np.array(res_config[i]).sum()
        res_config[i].append(num_res - other_source)
    return res_config


if __name__ =='__main__':
    print(gen_configs_recursively_fix(5,3))