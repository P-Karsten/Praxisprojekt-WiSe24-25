
import subprocess


directory = r"C:\Users\Pascal\Documents\VsCode Projects\Praxisprojekt-WiSe24-25\src\main\kotlin\cga\exercise\game\py"


commands = [
    f"cd /d {directory} && tensorboard --logdir=logs/game_rewards/",
    f"cd /d {directory} && python -m uvicorn test:app",
    #f"cd /d {directory} && python GameEnv.py"
]


wt_command = "wt"
for cmd in commands:
    wt_command += f" new-tab cmd /k \"{cmd}\" ;"


subprocess.run(wt_command, shell=True)