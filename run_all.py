import subprocess
import time
import sys
import psutil

def kill_process_tree(pid):
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass

print("Запуск сервера...")
server_proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "server:app", "--reload"]
)
time.sleep(5)

print("Запуск клиента...")
try:
    subprocess.call([sys.executable, "-m", "streamlit", "run", "client.py"])
finally:
    print("Остановка сервера...")
    kill_process_tree(server_proc.pid)
