import os
import subprocess
import time
from tqdm import tqdm
# 指定主目录
main_directory = '/mnt/petrelfs/limo/opencompass_fork/configs/needleinahaystack/multi'

# 遍历主目录中的所有子目录
for subdir in os.listdir(main_directory):
    subdir_path = os.path.join(main_directory, subdir)
    if os.path.isdir(subdir_path):
        # 列出子目录下所有 Python 文件
        files = [f for f in os.listdir(subdir_path) if f.endswith('.py')]
        # 依次执行每个文件
        for file in tqdm(files):
            full_path = os.path.join(subdir_path, file)
            command = f'python run_tmux.py {full_path}'
            subprocess.run(command, shell=True)
            time.sleep(1)
