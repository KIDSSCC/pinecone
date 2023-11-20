import subprocess as sp

if __name__ == '__main__':
    print('prepare to test redis')
    test_command = f"redis-benchmark -c 50 -n 200"
    test_process = sp.Popen(test_command, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
    test_process.wait()
    print('test end')
    stdout, stderr = test_process.communicate()
    print("Standard Output:\n", stdout.decode())
    print(type(stdout))