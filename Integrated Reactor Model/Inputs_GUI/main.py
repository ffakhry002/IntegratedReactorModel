# Main entry point for the reactor GUI application
#!/usr/bin/env python3
"""
Main entry point for the Interactive Reactor Design GUI
"""
import tkinter as tk
import sys
import os

# Add parent directory to path to import reactor modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Inputs_GUI.reactor_gui import ReactorGUI


def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = ReactorGUI(root)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nGUI closed by user")


if __name__ == "__main__":
    main()
