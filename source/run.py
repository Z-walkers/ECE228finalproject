import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
import sys


TASKS = {
    "burgers": {
        "pinn": "source/Burgers1D_PINN.py",
        "stpinn": "source/Burgers1D_STPINN.py",
        "q_max": 0.2,
    },
    "diffreact": {
        "pinn": "source/DiffReact1D_PINN.py",
        "stpinn": "source/DiffReact1D_STPINN.py",
        "q_max": 0.5,
    },
    "diffsorb": {
        "pinn": "source/DiffSorb1D_PINN.py",
        "stpinn": "source/DiffSorb1D_STPINN.py",
        "q_max": 0.2,
    },
}


def build_command(script, method, task_config, adam_it, max_time, smoke=False):
    cmd = [sys.executable, script, "--adam-it", str(adam_it), "--max-time", str(max_time)]
    if smoke and "DiffReact" in script:
        cmd.append("--skip-lbfgs")
    if method == "stpinn":
        cmd.extend([
            "--variant", "stpinn",
            "--schedule-type", "fixed",
            "--q-max", str(task_config["q_max"]),
        ])
    elif method == "ffustpinn":
        cmd.extend([
            "--variant", "ffustpinn",
            "--schedule-type", "fixed",
            "--q-max", str(task_config["q_max"]),
        ])
    elif method == "dynamic-ffu":
        cmd.extend([
            "--variant", "ffustpinn",
            "--schedule-type", "linear_warmup",
            "--q-min", "0.02",
            "--q-max", str(task_config["q_max"]),
            "--warmup-ratio", "0.6",
        ])
    return cmd


def run_one(job):
    task, method, cmd = job
    header = "\n" + "=" * 72 + f"\nRunning task={task}, method={method}\n" + " ".join(cmd) + "\n" + "=" * 72
    print(header, flush=True)
    result = subprocess.run(cmd)
    return task, method, result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run 1D PINN/ST-PINN reproduction experiments.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["burgers", "diffreact", "diffsorb"],
        choices=list(TASKS.keys()),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["pinn", "stpinn", "ffustpinn"],
        choices=["pinn", "stpinn", "ffustpinn", "dynamic-ffu"],
    )
    parser.add_argument("--adam-it", type=int, default=20000)
    parser.add_argument("--max-time", type=float, default=10)
    parser.add_argument("--smoke", action="store_true", help="Run a short 300-iteration validation.")
    parser.add_argument("--parallel", action="store_true", help="Run selected experiments concurrently.")
    parser.add_argument("--workers", type=int, default=3, help="Number of concurrent experiments when --parallel is set.")
    args = parser.parse_args()

    adam_it = 300 if args.smoke else args.adam_it
    max_time = min(args.max_time, 0.25) if args.smoke else args.max_time

    jobs = []
    for task in args.tasks:
        for method in args.methods:
            script_key = "pinn" if method == "pinn" else "stpinn"
            task_config = TASKS[task]
            script = task_config[script_key]
            cmd = build_command(script, method, task_config, adam_it, max_time, smoke=args.smoke)
            jobs.append((task, method, cmd))

    if args.parallel:
        max_workers = max(1, min(args.workers, len(jobs)))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_one, job) for job in jobs]
            for future in as_completed(futures):
                task, method, returncode = future.result()
                if returncode != 0:
                    raise SystemExit(f"{task}/{method} failed with exit code {returncode}")
                print(f"Finished task={task}, method={method}", flush=True)
    else:
        for job in jobs:
            task, method, returncode = run_one(job)
            if returncode != 0:
                raise SystemExit(returncode)

    print("All selected experiments finished.")


if __name__ == "__main__":
    main()
