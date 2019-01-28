import os, sys
import hashlib
import threading, multiprocessing
from os.path import getsize
from tqdm import tqdm


class DiskWalk(object):
    def __init__(self, path):
        self.path = path
    
    def paths(self):
        path = self.path
        path_collection = []
        for dirpath, dirnames, filenames in os.walk(path):
            for file in filenames:
                fullpath = os.path.join(dirpath, file)
                path_collection.append(fullpath)
        return path_collection


def create_checksum(path):
    fp = open(path, 'rb')
    checksum = hashlib.md5()
    while True:
        buffer = fp.read(8192)
        if not buffer: break
        checksum.update(buffer)
    fp.close()
    checksum = checksum.digest()
    return checksum


def find_dupes(path):
    record = {}
    dup = {}
    d = DiskWalk(path)
    files = d.paths()
    for file in files:
        compound_key = (getsize(file), create_checksum(file))
        if compound_key in record:
            dup[file] = record[compound_key]
        else:
            record[compound_key] = file
    return dup


def deal(record, rt, folders):
    for folder in folders:
        rtf = os.path.join(rt, folder)
        files = os.listdir(rtf)
        for file in files:
            file_path = os.path.join(rtf, file)
            cc = create_checksum(file_path)
            if cc in record:
                os.remove(file_path)


def rm_none():
    files = os.listdir('./char_datasets')
    l = len(files)
    f2 = os.listdir('./labels')
    record = []
    for fx in f2:
        fxpath = os.path.join('./labels', fx)
        cc = create_checksum(fxpath)
        record.append(cc)

    for th in range(16):
        end_f = (th + 1) * 26
        if end_f > l:
            end_f = l
        file_used = files[th * 26 : end_f]
        t = threading.Thread(target=deal,
                             args=(record, './char_datasets/', file_used))
        t.start()


if __name__ == '__main__':
    rm_none()
