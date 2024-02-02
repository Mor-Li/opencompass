import argparse
import os
import subprocess


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_tmux_session(session_name, path, debug, reuse_path):
    subprocess.run(f'tmux new -s {session_name} -d', shell=True)
    subprocess.run(
        f"tmux send-keys -t {session_name} 'conda activate "
        "opencompass_fork' C-m",
        shell=True)

    debug_flag = '--debug' if debug else ''
    reuse_flag = f'-r {reuse_path}' if reuse_path else ''
    full_reuse_path = os.path.join(work_dir, reuse_path)
    ensure_directory_exists(full_reuse_path)
    command = f'python run.py {path} {debug_flag} {reuse_flag} --mode eval'
    subprocess.run(f"tmux send-keys -t {session_name} '{command}' C-m",
                   shell=True)


# Set up argument parsing
parser = argparse.ArgumentParser(
    description='Create a new tmux session and run a Python script.')
parser.add_argument('path', help='Path to the Python script')
parser.add_argument('--debug',
                    action='store_true',
                    help='Enable debug mode for the Python script')
parser.add_argument('-r', '--reuse', help='Reuse path for the Python script')

# Parse arguments
args = parser.parse_args()

# Generate session name and reuse path from the script filename
script_name = os.path.splitext(os.path.basename(args.path))[0]
session_name = script_name + '_eval'

reuse_path = args.reuse if args.reuse else script_name

work_dir = './outputs/needlebench'
ensure_directory_exists(work_dir)
ensure_directory_exists(os.path.join(work_dir, script_name))

# Main logic
create_tmux_session(session_name, args.path, args.debug, reuse_path)
