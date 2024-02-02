import re
import subprocess
from datetime import datetime, timedelta

from prettytable import PrettyTable


def get_tmux_sessions():
    try:
        output = subprocess.check_output('tmux ls', shell=True).decode()

    except subprocess.CalledProcessError:
        return []
    sessions = re.findall(r'^([^:]+):', output, re.MULTILINE)
    return sessions


def get_last_lines_of_session(session_name, num_lines=10):
    try:
        output = subprocess.check_output(
            f'tmux capture-pane -t {session_name} -p -S -{num_lines}',
            shell=True).decode()
        print(session_name)
        print(output)
        lines = output.strip().split('\n')
        return '\n'.join(line.rstrip() for line in lines[-num_lines:])
    except subprocess.CalledProcessError:
        return 'No content available or capture failed.'


def extract_progress_info(last_lines):
    # 仅匹配 "Finished" 行的进度信息
    progress_match = re.search(r'Finished:\s+(\d+)%\|', last_lines)
    if progress_match:
        progress = progress_match.group(1).strip()

        # 匹配 "Finished" 行中的时间信息
        time_match = re.search(r'Finished:.*?\[(\d+:\d+:\d+|\d+:\d+)<.*?\]',
                               last_lines)
        if time_match:
            time_parts = list(map(int, time_match.group(1).split(':')))
            if len(time_parts) == 2:
                time_parts.insert(0, 0)  # 前面添加0小时
            h, m, s = time_parts

            # 根据已用时间和完成百分比估计总时间
            if progress != '0':  # 防止除以零
                total_time_sec = int(h * 3600 + m * 60 + s) / (int(progress) /
                                                               100)
                remaining_time_sec = total_time_sec - (h * 3600 + m * 60 + s)
                estimated_completion_time = datetime.now() + timedelta(
                    seconds=remaining_time_sec)
                pt = estimated_completion_time.strftime('%Y-%m-%d %H:%M:%S')
                return (f'Incomplete ({progress}%) - Est. Completion: '
                        f'{pt}')
            else:
                return 'Incomplete (0%) - Cannot estimate time'
    return 'No progress info'


def extract_time_info(time_match, progress):
    time_parts = list(map(int, time_match.group(1).split(':')))
    if len(time_parts) == 2:
        time_parts.insert(0, 0)  # 前面添加0小时
    h, m, s = time_parts

    estimated_completion_time = datetime.now() + timedelta(
        hours=h, minutes=m, seconds=s)
    return (f'Incomplete ({progress}%) - Est. Completion: '
            f'{estimated_completion_time.strftime("%Y-%m-%d %H:%M:%S")}')


def check_session_status(session_name):
    last_lines = get_last_lines_of_session(session_name)
    return 'write' in last_lines, extract_progress_info(last_lines)


# Main logic
sessions = get_tmux_sessions()
session_statuses = []

for session in sessions:
    completed, progress_info = check_session_status(session)
    status = 'Completed' if completed else progress_info
    session_statuses.append((session, status, completed))

# Sort sessions first by completion status, then by progress percentage
session_statuses.sort(
    key=lambda x:
    (x[2], -int(re.search(r'(\d+)%', x[1]).group(1))
     if 'Incomplete' in x[1] and 'No progress info' not in x[1] else 0),
    reverse=True)

# Print table with session statuses
table = PrettyTable(['Session Name', 'Status'])
for session, status, _ in session_statuses:
    table.add_row([session, status])
print(table)
