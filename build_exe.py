import os
import shutil

os.system('pyinstaller run.py --noconsole --onefile')
os.remove('run.exe')
shutil.move('dist/run.exe', 'run.exe')
shutil.rmtree('build')
shutil.rmtree('dist')
os.remove('run.spec')