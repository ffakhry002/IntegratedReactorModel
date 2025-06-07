#!/usr/bin/env python3
"""
Utility to view and manage ML training logs
"""

import os
import sys
import glob
from datetime import datetime

def list_logs(logs_dir):
    """List all log files with their sizes and dates"""
    log_files = glob.glob(os.path.join(logs_dir, "ml_training_*.log"))

    if not log_files:
        print("No log files found.")
        return []

    # Sort by modification time (newest first)
    log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    print(f"\n{'='*80}")
    print(f"{'Filename':<40} {'Size':>10} {'Modified':<20}")
    print(f"{'='*80}")

    for log_file in log_files:
        filename = os.path.basename(log_file)
        size = os.path.getsize(log_file)
        mtime = datetime.fromtimestamp(os.path.getmtime(log_file))

        # Format size
        if size > 1024*1024:
            size_str = f"{size/(1024*1024):.1f} MB"
        elif size > 1024:
            size_str = f"{size/1024:.1f} KB"
        else:
            size_str = f"{size} B"

        print(f"{filename:<40} {size_str:>10} {mtime.strftime('%Y-%m-%d %H:%M:%S'):<20}")

    print(f"{'='*80}")
    print(f"Total: {len(log_files)} log files")

    return log_files

def view_log(log_file, lines=50, grep=None):
    """View the last N lines of a log file, optionally grep for patterns"""
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return

    with open(log_file, 'r') as f:
        all_lines = f.readlines()

    if grep:
        # Filter lines containing the pattern
        filtered_lines = [line for line in all_lines if grep.lower() in line.lower()]
        if filtered_lines:
            print(f"\nLines containing '{grep}' ({len(filtered_lines)} matches):")
            print("="*80)
            for line in filtered_lines[-lines:]:
                print(line.rstrip())
        else:
            print(f"No lines found containing '{grep}'")
    else:
        # Show last N lines
        print(f"\nLast {lines} lines of {os.path.basename(log_file)}:")
        print("="*80)
        for line in all_lines[-lines:]:
            print(line.rstrip())

def search_errors(log_file):
    """Search for errors, warnings, and timeouts in log file"""
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return

    patterns = {
        'ERRORS': ['error:', 'exception', 'failed', 'traceback'],
        'WARNINGS': ['warning', 'timeout', '[WARNING]', '[TIMEOUT]'],
        'BEST SCORES': ['[NEW BEST]', 'best score:', 'best params:'],
        'COMPLETION': ['complete!', 'finished', 'saved to:']
    }

    with open(log_file, 'r') as f:
        lines = f.readlines()

    for category, keywords in patterns.items():
        matches = []
        for i, line in enumerate(lines):
            if any(keyword.lower() in line.lower() for keyword in keywords):
                matches.append((i, line.rstrip()))

        if matches:
            print(f"\n{category} ({len(matches)} found):")
            print("-" * 80)
            for line_num, line in matches[-10:]:  # Show last 10
                print(f"Line {line_num + 1}: {line}")

def get_summary(log_file):
    """Extract summary information from log file"""
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return

    with open(log_file, 'r') as f:
        content = f.read()

    print(f"\nLog Summary for {os.path.basename(log_file)}:")
    print("="*80)

    # Extract key information
    info = {
        'Start Time': None,
        'End Time': None,
        'Models Trained': [],
        'Best Scores': [],
        'Errors': 0,
        'Timeouts': 0
    }

    lines = content.split('\n')
    for line in lines:
        if 'ML Training Session Started:' in line:
            info['Start Time'] = line.split('Started:')[1].strip()
        elif 'ML Training Session Completed' in line:
            info['End Time'] = "Completed successfully"
        elif 'Training cancelled by user' in line:
            info['End Time'] = "Cancelled by user"
        elif 'Training' in line and 'for' in line and any(model in line.upper() for model in ['XGBOOST', 'RANDOM_FOREST', 'SVM', 'NEURAL_NET']):
            info['Models Trained'].append(line.strip())
        elif '[NEW BEST]' in line:
            info['Best Scores'].append(line.strip())
        elif 'error:' in line.lower() or 'exception' in line.lower():
            info['Errors'] += 1
        elif 'timeout' in line.lower():
            info['Timeouts'] += 1

    # Print summary
    print(f"Start Time: {info['Start Time'] or 'Not found'}")
    print(f"End Time: {info['End Time'] or 'Still running or crashed'}")
    print(f"Models Trained: {len(info['Models Trained'])}")
    for model in info['Models Trained'][:5]:  # Show first 5
        print(f"  - {model}")
    if len(info['Models Trained']) > 5:
        print(f"  ... and {len(info['Models Trained']) - 5} more")

    print(f"\nErrors: {info['Errors']}")
    print(f"Timeouts: {info['Timeouts']}")

    if info['Best Scores']:
        print(f"\nBest Scores Found: {len(info['Best Scores'])}")
        for score in info['Best Scores'][-3:]:  # Show last 3
            print(f"  {score}")

def main():
    """Main entry point for log viewer"""
    # Get ML directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ml_dir = os.path.dirname(script_dir)
    logs_dir = os.path.join(ml_dir, "outputs", "logs")

    if not os.path.exists(logs_dir):
        print(f"Logs directory not found: {logs_dir}")
        return

    print("\nML Training Log Viewer")
    print("="*80)

    while True:
        print("\nOptions:")
        print("1. List all log files")
        print("2. View latest log")
        print("3. View specific log")
        print("4. Search for errors/warnings")
        print("5. Get log summary")
        print("6. Search for pattern")
        print("0. Exit")

        choice = input("\nSelect option: ").strip()

        if choice == '0':
            break
        elif choice == '1':
            list_logs(logs_dir)
        elif choice == '2':
            log_files = glob.glob(os.path.join(logs_dir, "ml_training_*.log"))
            if log_files:
                latest = max(log_files, key=os.path.getmtime)
                view_log(latest)
            else:
                print("No log files found.")
        elif choice == '3':
            log_files = list_logs(logs_dir)
            if log_files:
                idx = input("\nEnter log number (1-{}): ".format(len(log_files))).strip()
                try:
                    idx = int(idx) - 1
                    if 0 <= idx < len(log_files):
                        lines = input("How many lines to show (default 50): ").strip()
                        lines = int(lines) if lines else 50
                        view_log(log_files[idx], lines)
                    else:
                        print("Invalid selection.")
                except ValueError:
                    print("Invalid input.")
        elif choice == '4':
            log_files = glob.glob(os.path.join(logs_dir, "ml_training_*.log"))
            if log_files:
                latest = max(log_files, key=os.path.getmtime)
                search_errors(latest)
            else:
                print("No log files found.")
        elif choice == '5':
            log_files = glob.glob(os.path.join(logs_dir, "ml_training_*.log"))
            if log_files:
                latest = max(log_files, key=os.path.getmtime)
                get_summary(latest)
            else:
                print("No log files found.")
        elif choice == '6':
            log_files = glob.glob(os.path.join(logs_dir, "ml_training_*.log"))
            if log_files:
                latest = max(log_files, key=os.path.getmtime)
                pattern = input("Enter search pattern: ").strip()
                if pattern:
                    view_log(latest, lines=50, grep=pattern)
            else:
                print("No log files found.")
        else:
            print("Invalid option.")

if __name__ == "__main__":
    main()
