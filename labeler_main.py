if __name__ == '__main__':
    from check_env import check_env

    check_env("labeler")
    from ant_tracker.labeler.AntLabeler import main

    main()
