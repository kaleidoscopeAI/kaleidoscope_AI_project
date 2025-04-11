#!/usr/bin/env python3
"""
Quantum Kaleidoscope Launcher
=============================

A launcher script for the Enhanced Quantum Kaleidoscope System.
This launches both the main system and the advanced visualizer.
"""

import os
import sys
import time
import subprocess
import threading
import argparse
import signal
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("KaleidoscopeLauncher")

class ProcessManager:
    def __init__(self):
        self.processes = {}
        self.stop_event = threading.Event()
    
    def start_process(self, name, cmd, cwd=None):
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd
            )
            self.processes[name] = process
            threading.Thread(
                target=self._monitor_process_output,
                args=(name, process),
                daemon=True
            ).start()
            logger.info(f"Started process {name} (PID: {process.pid})")
            return True
        except Exception as e:
            logger.error(f"Error starting process {name}: {e}")
            return False
    
    def _monitor_process_output(self, name, process):
        while not self.stop_event.is_set():
            if process.poll() is not None:
                if process.returncode != 0:
                    logger.error(f"Process {name} terminated with code {process.returncode}")
                else:
                    logger.info(f"Process {name} terminated normally")
                break
            line = process.stdout.readline()
            if line:
                logger.info(f"[{name}] {line.strip()}")
            line = process.stderr.readline()
            if line:
                logger.error(f"[{name}] {line.strip()}")
            time.sleep(0.1)
    
    def stop_all(self):
        self.stop_event.set()
        for name, process in self.processes.items():
            logger.info(f"Stopping process {name} (PID: {process.pid})")
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(f"Process {name} did not terminate gracefully, forcing...")
                process.kill()
            except Exception as e:
                logger.error(f"Error stopping process {name}: {e}")
        self.processes.clear()

def get_script_path(script_name):
    launcher_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(launcher_dir, script_name)

def launch_system(data_dir, main_port, visualizer_port, auto_gen):
    logger.info("Starting Quantum Kaleidoscope System...")
    process_mgr = ProcessManager()
    os.makedirs(data_dir, exist_ok=True)
    
    main_script = get_script_path("enhanced-quantum-kaleidoscope.py")
    visualizer_script = get_script_path("quantum_visualizer_new.py")  # Updated to new script
    
    # Launch main system
    main_cmd = [
        sys.executable,
        main_script,
        "--data-dir", data_dir,
        "--port", str(main_port)
    ]
    if auto_gen:
        main_cmd.append("--auto-gen")
    
    if not process_mgr.start_process("main", main_cmd):
        logger.error("Failed to start main system")
        return None
    
    logger.info("Waiting for main system to initialize...")
    time.sleep(3)
    
    # Launch visualizer
    visualizer_cmd = [
        sys.executable,
        visualizer_script,
        "--api-port", str(main_port),
        "--port", str(visualizer_port)
    ]
    
    if not process_mgr.start_process("visualizer", visualizer_cmd):
        logger.error("Failed to start visualizer")
    
    logger.info(f"""
===========================================================
Quantum Kaleidoscope System Started Successfully!
===========================================================
Main System URL: http://localhost:{main_port}
Advanced Visualizer URL: http://localhost:{visualizer_port}
===========================================================
Press Ctrl+C to stop the system
===========================================================
""")
    return process_mgr

def main():
    parser = argparse.ArgumentParser(description="Quantum Kaleidoscope Launcher")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--main-port", type=int, default=8000, help="Port for main system")
    parser.add_argument("--visualizer-port", type=int, default=8081, help="Port for visualizer")
    parser.add_argument("--auto-gen", action="store_true", help="Enable auto-generation")
    
    args = parser.parse_args()
    
    process_mgr = launch_system(
        data_dir=args.data_dir,
        main_port=args.main_port,
        visualizer_port=args.visualizer_port,
        auto_gen=args.auto_gen
    )
    
    if not process_mgr:
        logger.error("Failed to launch system")
        sys.exit(1)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down Quantum Kaleidoscope System...")
        process_mgr.stop_all()
        logger.info("System shutdown complete")
        sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(0))
    main()
