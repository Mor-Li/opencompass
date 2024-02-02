import os
import subprocess
import time

# 指定目录
directory = 'configs/needlebench/run_needledatacompare_configs'

# 列出目录下所有 Python 文件
files = [f for f in os.listdir(directory) if f.endswith('.py')]

# 依次执行每个文件
for file in files:
    full_path = os.path.join(directory, file)
    command = f'python run_tmux.py {full_path}'
    subprocess.run(command, shell=True)
    time.sleep(0.5)
