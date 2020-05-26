import os


def get_project_root_path():
    return os.path.dirname(os.path.realpath(__file__))


if __name__ == '__main__':
    print(get_project_root_path())
