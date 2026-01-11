import subprocess, sys, time, statistics

BOT = "battleship_bot.py"
STATE = "state_combat_force_sp.json"
RUNS = 10

times = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    subprocess.check_output([sys.executable, BOT, STATE])
    times.append(time.perf_counter() - t0)

print("runs:", RUNS)
print("min :", min(times))
print("mean:", statistics.mean(times))
print("p95 :", sorted(times)[int(0.95*(RUNS-1))])
print("max :", max(times))
