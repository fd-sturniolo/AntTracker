# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['labeler_main.py'],
             pathex=[],
             binaries=[],
             datas=[
                 ('./ant_tracker/labeler/style.css', '.'),
                 ('./ant_tracker/labeler/leaf.png', '.'),
             ],
             hiddenimports=[],
             hookspath=['pyinstaller-hooks'],
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
          [],
          exclude_binaries=True,
          name='AntLabeler',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               options=[('v', None, 'OPTION')],
               strip=False,
               upx=True,
               upx_exclude=[],
               name='AntLabeler')
