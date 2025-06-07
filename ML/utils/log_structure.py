"""
Logging utilities for ML training system
Captures EXACTLY what appears in the terminal - nothing more, nothing less
"""

import sys
import os
import fcntl
import select
import threading
from datetime import datetime


class TerminalLogger:
    """Logger that captures exactly what appears in the terminal"""

    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.log_file = open(log_file_path, 'a', buffering=1)  # Line buffered
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.stdout_fd = sys.stdout.fileno()
        self.stderr_fd = sys.stderr.fileno()

        # Save original file descriptors
        self.saved_stdout_fd = os.dup(self.stdout_fd)
        self.saved_stderr_fd = os.dup(self.stderr_fd)

        # Create pipes for capturing output
        self.stdout_pipe_r, self.stdout_pipe_w = os.pipe()
        self.stderr_pipe_r, self.stderr_pipe_w = os.pipe()

        # Make pipes non-blocking
        fcntl.fcntl(self.stdout_pipe_r, fcntl.F_SETFL, os.O_NONBLOCK)
        fcntl.fcntl(self.stderr_pipe_r, fcntl.F_SETFL, os.O_NONBLOCK)

        # Redirect stdout and stderr to pipes
        os.dup2(self.stdout_pipe_w, self.stdout_fd)
        os.dup2(self.stderr_pipe_w, self.stderr_fd)

        # Start reader threads
        self.stop_threads = False
        self.stdout_thread = threading.Thread(target=self._reader_thread,
                                            args=(self.stdout_pipe_r, self.saved_stdout_fd))
        self.stderr_thread = threading.Thread(target=self._reader_thread,
                                            args=(self.stderr_pipe_r, self.saved_stderr_fd))
        self.stdout_thread.daemon = True
        self.stderr_thread.daemon = True
        self.stdout_thread.start()
        self.stderr_thread.start()

    def _reader_thread(self, pipe_fd, console_fd):
        """Thread that reads from pipe and writes to both console and log file"""
        while not self.stop_threads:
            try:
                # Check if there's data to read
                ready, _, _ = select.select([pipe_fd], [], [], 0.1)
                if ready:
                    data = os.read(pipe_fd, 4096)
                    if data:
                        # Write to console (this is what the user sees)
                        os.write(console_fd, data)
                        # Write to log file (exact same content)
                        try:
                            self.log_file.write(data.decode('utf-8', errors='replace'))
                            self.log_file.flush()
                        except:
                            pass
            except OSError:
                # Pipe closed
                break
            except Exception:
                pass

    def close(self):
        """Restore original stdout/stderr and close log file"""
        # Stop reader threads
        self.stop_threads = True

        # Restore original file descriptors
        os.dup2(self.saved_stdout_fd, self.stdout_fd)
        os.dup2(self.saved_stderr_fd, self.stderr_fd)

        # Close pipes
        try:
            os.close(self.stdout_pipe_r)
            os.close(self.stdout_pipe_w)
            os.close(self.stderr_pipe_r)
            os.close(self.stderr_pipe_w)
            os.close(self.saved_stdout_fd)
            os.close(self.saved_stderr_fd)
        except:
            pass

        # Wait for threads to finish
        self.stdout_thread.join(timeout=1)
        self.stderr_thread.join(timeout=1)

        # Close log file
        self.log_file.close()


class SimpleLogger:
    """Fallback logger using simple Python stdout/stderr redirection"""

    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.stdout = TeeOutput(sys.stdout, log_file_path)
        self.stderr = TeeOutput(sys.stderr, log_file_path)
        sys.stdout = self.stdout
        sys.stderr = self.stderr

    def close(self):
        """Restore original stdout/stderr"""
        sys.stdout = self.stdout.terminal
        sys.stderr = self.stderr.terminal
        self.stdout.close()
        self.stderr.close()


class TeeOutput:
    """Helper class to duplicate output to both terminal and file"""

    def __init__(self, terminal, log_file_path):
        self.terminal = terminal
        self.log = open(log_file_path, 'a', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def setup_logging(outputs_dir):
    """
    Setup logging to capture exactly what appears in terminal
    Returns: (log_filepath, logger_object)
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(outputs_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create log filename
    log_filename = f"ml_training_{timestamp}.log"
    log_filepath = os.path.join(logs_dir, log_filename)

    # Set environment variables to ensure libraries output to stdout/stderr
    os.environ['PYTHONUNBUFFERED'] = '1'

    # Try to use terminal logger (captures C-level output), fallback to simple logger
    logger = None
    try:
        logger = TerminalLogger(log_filepath)
    except Exception as e:
        # Fallback to simple logger
        logger = SimpleLogger(log_filepath)

    # Log session start
    print(f"\n{'='*60}")
    print(f"ML Training Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_filepath}")
    print(f"{'='*60}\n")

    return log_filepath, logger


def close_logging(logger):
    """Properly close the logger"""
    if logger:
        logger.close()
