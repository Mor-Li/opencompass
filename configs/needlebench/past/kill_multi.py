import os
import subprocess
import time

# 指定主目录
main_directory = '/mnt/petrelfs/limo/opencompass_fork/configs/needleinahaystack/multi'

# 遍历主目录中的所有子目录
for subdir in os.listdir(main_directory):
    subdir_path = os.path.join(main_directory, subdir)
    if os.path.isdir(subdir_path):
        # 列出子目录下所有 Python 文件
        files = [f for f in os.listdir(subdir_path) if f.endswith('.py')]
        # 依次关闭每个文件对应的 tmux 会话
        for file in files:
            session_name = os.path.splitext(file)[0]  # 只获取文件名，去掉.py后缀
            command = f'tmux kill-session -t {session_name}'
            subprocess.run(command, shell=True)
            time.sleep(2)
