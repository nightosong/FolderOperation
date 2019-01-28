import os
import threading, multiprocessing

def file_capacity(file_path):
    s = os.path.getsize(file_path)
    return s

def file_read():
    test_path = './char_datasets/0409/0023.png'
    file_cap = file_capacity(test_path)
    print(type(file_cap), file_cap)

def remove_files(file_root, file_list):
    for f in file_list:
        s = os.path.join(file_root, f)
        files = os.listdir(s)
        for x in files:
            sx = os.path.join(s, x)
            if fileCapacity(sx) <= 148:
                os.remove(sx)

if __name__ == '__main__':
    files = os.listdir('./char_datasets')
    l = len(files)
    for th in range(16):
        end_f = (th + 1) * 26
        if end_f > l:
            end_f = l
        file_used = files[th * 26 : end_f]
        t = threading.Thread(target=remove_files,
                             args=('./char_datasets/', file_used))
        t.start()

