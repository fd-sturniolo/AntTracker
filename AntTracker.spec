# -*- mode: python ; coding: utf-8 -*-

import os

from pathlib import Path

os.environ['PATH'] = str(Path("lib").resolve()) + os.pathsep + os.environ['PATH']
print(os.environ['PATH'])

console = True
upx = False
upx_exclude = ['ucrtbase.dll', 'VCRUNTIME140.dll']

datas = [
    ('ant_tracker/tracker_gui/model.tflite', '.'),
    ('ant_tracker/tracker_gui/images', 'images'),
]

# binaries = [(dll, '.') for dll in glob.glob("lib/*.dll")]
binaries = []

block_cipher = None
a = Analysis(['tracker_main.py'],
             pathex=[],
             binaries=binaries,
             datas=datas,
             hiddenimports=['pkg_resources.py2_warn'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=True)

pyz = PYZ(a.pure, a.zipped_data,
          cipher=block_cipher)

# # one-file mode
# exe = EXE(pyz,
#           a.scripts,
#           a.binaries,
#           a.zipfiles,
#           a.datas,
#           # [],
#           name='AntTracker',
#           debug=False,
#           bootloader_ignore_signals=False,
#           strip=False,
#           upx=upx,
#           upx_exclude=upx_exclude,
#           runtime_tmpdir=None,
#           console=console)

# folder mode
exe = EXE(pyz,
          a.scripts,
          # [('v', None, 'OPTION')],  # Verbose output
          exclude_binaries=True,
          name='AntTracker',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=upx,
          console=console,
          icon="ant_tracker/tracker_gui/images/icon.ico")

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=upx,
               upx_exclude=upx_exclude,
               name='AntTracker')
