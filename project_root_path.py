import os
import sys


def get_project_root_path():
    return os.path.dirname(os.path.realpath(__file__))


def set_project_path():
    root_path = get_project_root_path()
    if root_path not in sys.path:
        sys.path.append(get_project_root_path())


if __name__ == '__main__':
    print(get_project_root_path())
