from project_root_path import set_project_path


def main():
    set_project_path()

    from src.NeuralNetwork.Training.train import main as train
    train()


if __name__ == '__main__':
    main()
