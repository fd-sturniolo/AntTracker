def check_env():
    import sys
    frozen = getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')
    if not frozen:
        from pathlib import Path
        for file in (Path("../.env_info"), Path(".env_info")):
            env_info_file = file
            if env_info_file.exists():
                break
        else:
            raise ValueError("Debe generar un conda env con create-env.ps1")
        needed_env = [line.split(':')[1] for line in env_info_file.read_text().split("\n") if line.startswith('tracker')]
        if not needed_env:
            raise ValueError("Debe generar un conda env con create-env.ps1")
        needed_env = needed_env[0]
        import os
        current_env = os.environ['CONDA_DEFAULT_ENV']
        if needed_env != current_env:
            raise ValueError(f"Sólo ejecutar este archivo en el conda-env "
                             f"generado por create-env.ps1 ({needed_env}).\n"
                             f"Actualmente está utilizando: {current_env}")
