import subprocess
import argparse
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Directory where data is located', type=str)
    args, _ = parser.parse_known_args()

    # Call the new CLI
    print(f"Forwarding to new pipeline: evaluation process --benchmark-dir {args.path} --legacy-csv")
    try:
        subprocess.run(["evaluation", "process", "--benchmark-dir", args.path, "--legacy-csv"], check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("Error: 'evaluation' command not found. Have you run 'pip install -e .' in the arena_evaluation package?")
        sys.exit(1)