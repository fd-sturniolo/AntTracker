from PyInstaller.utils.hooks import collect_data_files

hiddenimports = ["pyforms.settings", "pyforms_gui", "pyforms_gui.settings", "pyforms.controls", "pyforms_gui.resources_settings", "pyforms.resources_settings"]

datas = collect_data_files('pyforms',include_py_files=True)
datas = collect_data_files('pyforms_gui',include_py_files=True)

print("------ Hooked pyforms ------")
