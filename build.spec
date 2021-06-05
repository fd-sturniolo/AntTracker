# -*- mode: python ; coding: utf-8 -*-

block_cipher = None
console = True
interpreter_options = [
    # ('v', None, 'OPTION'), #* Verbose output
]
pyinstaller_debug = False


tracker_a = Analysis(
    ['tracker_main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('ant_tracker/tracker_gui/model.tflite', '.'),
        ('ant_tracker/tracker_gui/images', 'images'),
    ],
    hiddenimports=['pkg_resources.py2_warn'],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=True)
labeler_a = Analysis(
    ['labeler_main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('./ant_tracker/labeler/style.css', '.'),
        ('./ant_tracker/labeler/leaf.png', '.')
    ],
    hiddenimports=[],
    hookspath=['pyinstaller-hooks'],
    runtime_hooks=[],
    excludes=['FixTk', 'tcl', 'tk', '_tkinter', 'tkinter', 'Tkinter'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False)

MERGE(
    (tracker_a, 'tracker_main', 'AntTracker'),
    (labeler_a, 'labeler_main', 'AntLabeler'),
)

tracker_pyz = PYZ(tracker_a.pure, tracker_a.zipped_data, cipher=block_cipher)
tracker_exe = EXE(
    tracker_pyz,
    tracker_a.scripts,
    interpreter_options,
    exclude_binaries=True,
    name='AntTracker',
    debug=pyinstaller_debug,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=console,
    icon="ant_tracker/tracker_gui/images/icon.ico")

labeler_pyz = PYZ(labeler_a.pure, labeler_a.zipped_data, cipher=block_cipher)
labeler_exe = EXE(
    labeler_pyz,
    labeler_a.scripts,
    interpreter_options,
    exclude_binaries=True,
    name='AntLabeler',
    debug=pyinstaller_debug,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=console)

COLLECT(
    labeler_exe,
    labeler_a.binaries,
    labeler_a.zipfiles,
    labeler_a.datas,
    tracker_exe,
    tracker_a.binaries,
    tracker_a.zipfiles,
    tracker_a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='AntTracker')
