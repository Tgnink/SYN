# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['synapse.py'],
<<<<<<< HEAD
             pathex=['E:\\PROJECT\\python\\SYN history\\version4_clean'],
=======
             pathex=['E:\\PROJECT\\python\\SYNS'],
>>>>>>> 33fb47138e32dbfa6c18e6c2e63781a8e1fa5c01
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
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
          name='synapse',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
<<<<<<< HEAD
          console=True , icon='synapse.ico')
=======
          console=True )
>>>>>>> 33fb47138e32dbfa6c18e6c2e63781a8e1fa5c01
