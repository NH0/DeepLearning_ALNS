from project_root_path import set_project_path


def main():
    set_project_path()

    from src.ALNS.AlnsStatistics.generate_alns_stats import generate_stats
    generate_stats()


if __name__ == '__main__':
    main()
