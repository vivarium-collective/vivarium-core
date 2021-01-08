import os


def open_relative(path, flag):
    relative = os.path.join(os.path.dirname(__file__), '../..', path)
    return open(relative, flag)
