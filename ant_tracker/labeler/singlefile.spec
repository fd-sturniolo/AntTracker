# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['AntLabeler.py'],
             pathex=['.'],
             binaries=[],
             datas=[('./style.css','.'),
                    ('./leaf.png','.'),
                    ('./Qt5Core.dll','PyQt5/Qt/bin'),
                    ('./Qt5Gui.dll','PyQt5/Qt/bin'),
                    ('./Qt5Widgets.dll','PyQt5/Qt/bin'),
                    ('./opencv_videoio_ffmpeg411_64.dll','.'),
					],
             hiddenimports=[],
             hookspath=['.'],
             runtime_hooks=[],
             excludes=['FixTk', 'tcl', 'tk', '_tkinter', 'tkinter', 'Tkinter'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='AntLabeler',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False )
