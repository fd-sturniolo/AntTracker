if __name__ == '__main__':
    from check_env import check_env

    check_env()
    from ant_tracker.tracker_gui.trkviz import trkviz as main

    main()
