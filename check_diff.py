import argparse
import json
from termcolor import colored

def load_json(path):
    """Load a JSON file and return its content."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Compare horse club fan statistics between two JSON files.")
    parser.add_argument("--previous", "-p", required=True, help="Path to previous week JSON file.")
    parser.add_argument("--current", "-c", required=True, help="Path to current week JSON file.")
    parser.add_argument("--threshold", type=float, required=True, help="Fan increase threshold for green highlight.")
    args = parser.parse_args()

    previous_data = load_json(args.previous)
    current_data = load_json(args.current)

    prev_dict = {member["name"]: member for member in previous_data}
    curr_dict = {member["name"]: member for member in current_data}

    diffs = []

    for name, curr in curr_dict.items():
        if name not in prev_dict:
            print(colored(f"[NEW MEMBER] {name}", "yellow"))
            continue

        old = prev_dict[name]
        diff = curr["total_fans"] - old["total_fans"]

        color = "green" if diff >= args.threshold else "red"
        diffs.append((name, diff, color))

    # Sort by diff descending
    diffs.sort(key=lambda x: x[1], reverse=True)

    print("\n=== Fan Growth Report ===")
    for name, diff, color in diffs:
        sign = "+" if diff >= 0 else ""
        print(colored(f"{name}: {sign}{diff:,}", color))

if __name__ == "__main__":
    main()