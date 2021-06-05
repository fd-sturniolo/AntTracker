from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    from check_env import check_env

    check_env()
    from ant_tracker.tracker_gui.main_window import main

    main()
