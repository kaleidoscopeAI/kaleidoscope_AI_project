# Corrected integrate_kaleidoscope.py
import os
import sys
import shutil
import logging
import argparse
import re
import platform # Import platform here for use in copy_and_rename
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("KaleidoscopeIntegrator")

# --- File Definitions (Based on Uploaded Files) ---

# Prioritized list of sources for each target file
TARGET_FILES = {
    "unified_quantum_kaleidoscope.py": [
        "enhanced-quantum-kaleidoscope.py", # Preferred name based on integration scripts
        "quantum-swarm-network(3).py",      # Seemingly most advanced core
        "unified_quantum_kaleidoscope(3).py",# Try numbered versions
        "unified_quantum_kaleidoscope(2).py",
        "unified_quantum_kaleidoscope(1).py",
        "unified_quantum_kaleidoscope.py",  # Original unified name
        "quantum_kaleidoscope_core.py",     # Core logic variations
        "quantum_kaleidoscope_core(1).py",
    ],
    "quantum_visualizer_new.py": [
        "quantum_visualizer_new.py",        # Expected by launcher
        "visualization-renderer.py",        # Alternative Flask renderer
        "visualization-renderer(1).py",
        "quantum_visualization.py",         # Original Flask visualizer
        "quantum_visualization(1).txt",     # Text versions
        "quantum_visualization(2).txt",
        "quantum_visualization.txt",
        "qsin-network-visualizer.py",       # QSIN specific, might be different
    ],
    "quantum_kaleidoscope_launcher.py": [
        "quantum_kaleidoscope_launcher.py",
        "quantum_kaleidoscope_launcher(1).txt",
        "quantum_kaleidoscope_launcher.txt",
    ],
    "visualization_integration.py": [ # Need this for patching logic
        "visualization-integration.py",
        "quantum-integrator.py",         # May contain similar logic
        "quantum_integration_manager.py" # May contain similar logic
    ]
}

# Essential modules to copy directly (if they exist)
ESSENTIAL_MODULES = [
    "quantum-server.py",
    "quantum-client.py",
    "supernode-manager.py",
    "supernode-processor.py",
    "quantum-sync-protocol.py",
    "kaleidoscope-integration.py", # High-level integrator
    # Add other core modules identified from imports if needed
    # e.g., "app_generator.py", "error_handling.py", "core_reconstruction.py", etc.
    # "llm_integration.py" # Assuming Untitled-10/11 are merged into this
]

# Static/Template files
WEB_ASSETS = {
    "templates": [
        "quantum-kaleidoscope.html",
        "quantum-kaleidoscope(1).html",
        "advanced-cube.html" # Choose one or merge for index.html/visualization.html later
    ],
    "static/js": [
        "quantum-kaleidoscope-frontend.js",
        # Add api.js, visualizer.js if generated/expected by unified system
    ],
    "static/css": [
        # Add style.css if generated/expected
    ]
}

# --- Helper Functions ---

def find_best_source(source_files: List[str], source_dir: str) -> Optional[str]:
    """Find the best existing source file from a prioritized list."""
    for fname in source_files:
        fpath = os.path.join(source_dir, fname)
        if os.path.exists(fpath):
            logger.debug(f"Found source: {fname}")
            return fpath
    logger.warning(f"Could not find any suitable source from list: {source_files}")
    return None

def copy_and_rename(source_path: str, target_filename: str, target_dir: str):
    """Copy the source file to the target directory with the target name."""
    if not source_path:
        logger.error(f"Cannot copy source for {target_filename}, no source found.")
        return False
    try:
        target_path = os.path.join(target_dir, target_filename)
        shutil.copy2(source_path, target_path)
        logger.info(f"Copied {os.path.basename(source_path)} to {target_filename}")
        # Make executable (important for launchers/scripts)
        if platform.system() != 'Windows' and target_filename.endswith(".py"):
             os.chmod(target_path, 0o755)
        return True
    except Exception as e:
        logger.error(f"Error copying {source_path} to {target_path}: {e}")
        return False

def create_directories(target_dir: str):
    """Create necessary directories in the target location."""
    try:
        os.makedirs(target_dir, exist_ok=True)
        os.makedirs(os.path.join(target_dir, "static", "js"), exist_ok=True)
        os.makedirs(os.path.join(target_dir, "static", "css"), exist_ok=True)
        os.makedirs(os.path.join(target_dir, "templates"), exist_ok=True)
        os.makedirs(os.path.join(target_dir, "data"), exist_ok=True) # For data persistence
        logger.info(f"Created necessary directories in {target_dir}")
    except Exception as e:
        logger.error(f"Error creating directories: {e}")
        raise

def apply_visualization_patch(target_system_path: str, integration_module_path: str):
    """
    Attempts to automatically patch the target system script to use the
    visualization integration logic.

    NOTE: This is a best-effort attempt and might require manual adjustment.
    It assumes the target system creates an instance named 'system' or 'app.system'
    and that the patching function is named 'patch_visualization_system'.
    """
    if not os.path.exists(target_system_path):
        logger.error(f"Target system file for patching not found: {target_system_path}")
        return False
    if not os.path.exists(integration_module_path):
        logger.error(f"Integration module for patching not found: {integration_module_path}")
        return False

    logger.info(f"Attempting to patch {os.path.basename(target_system_path)} for visualization...")

    try:
        with open(target_system_path, 'r') as f:
            lines = f.readlines()

        integration_module_name = os.path.basename(integration_module_path).replace('.py', '')
        patch_import = f"from {integration_module_name} import patch_visualization_system\n"
        patch_call = "    patch_visualization_system(system) # Auto-patched by integrator\n"
        alternate_patch_call = "    patch_visualization_system(app.system) # Auto-patched by integrator\n"

        # Check if already patched
        if any(patch_import.strip() in line for line in lines) and \
           any(patch_call.strip() in line or alternate_patch_call.strip() in line for line in lines):
            logger.info("System seems already patched.")
            return True

        # Find import location
        import_insert_line = 0
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                import_insert_line = i + 1
            elif line.strip() and not line.startswith("#"): # Stop after first non-import, non-comment line
                break

        # Find patch call location (look for system instantiation or app run)
        patch_call_insert_line = -1
        system_var_name = "system" # Default guess
        for i, line in enumerate(lines):
             # Detect common system instantiation patterns
            if re.search(r"^\s*(\w+)\s*=\s*QuantumKaleidoscope\(", line):
                 system_var_name = line.split("=")[0].strip()
                 patch_call = f"    patch_visualization_system({system_var_name}) # Auto-patched by integrator\n"
                 patch_call_insert_line = i + 1
                 break
            elif re.search(r"^\s*app\s*=\s*create_app\(", line):
                 system_var_name = "app.system" # Assume system is attached to app
                 patch_call = f"    patch_visualization_system({system_var_name}) # Auto-patched by integrator\n"
                 # Insert before app.run() or similar
                 continue # Keep searching for the run call
            elif re.search(r"^\s*app\.run\(", line) and patch_call_insert_line == -1:
                 # If we found app creation earlier, insert before run
                 if system_var_name == "app.system":
                      patch_call_insert_line = i
                 break # Stop searching after finding app.run

        if patch_call_insert_line == -1:
            # Fallback: try inserting near the end, before main execution block
            for i in range(len(lines) - 1, 0, -1):
                if lines[i].strip() and not lines[i].startswith(" "): # Found a non-indented line likely near end
                    patch_call_insert_line = i
                    break
            if patch_call_insert_line == -1: patch_call_insert_line = len(lines) # Append if desperate

        # Insert lines
        modified_lines = lines[:import_insert_line] + [patch_import] + lines[import_insert_line:]
        # Adjust insert line index due to previous insertion
        if patch_call_insert_line >= import_insert_line: patch_call_insert_line +=1

        modified_lines = modified_lines[:patch_call_insert_line] + [patch_call] + modified_lines[patch_call_insert_line:]


        # Write back the modified content
        with open(target_system_path, 'w') as f:
            f.writelines(modified_lines)

        logger.info(f"Successfully applied visualization patch to {os.path.basename(target_system_path)}")
        return True

    except Exception as e:
        logger.error(f"Error applying patch to {target_system_path}: {e}")
        logger.error(traceback.format_exc())
        return False

def copy_essential_modules(source_dir: str, target_dir: str):
    """Copy essential Python modules."""
    count = 0
    for module_name in ESSENTIAL_MODULES:
        source_path = os.path.join(source_dir, module_name)
        target_path = os.path.join(target_dir, module_name)
        if os.path.exists(source_path):
            try:
                shutil.copy2(source_path, target_path)
                logger.info(f"Copied essential module: {module_name}")
                count += 1
            except Exception as e:
                logger.warning(f"Could not copy essential module {module_name}: {e}")
        else:
             logger.debug(f"Essential module not found, skipping: {module_name}")
    logger.info(f"Copied {count} essential modules.")

def copy_web_assets(source_dir: str, target_dir: str):
    """Copy web assets (HTML, JS, CSS)."""
    copied_count = 0
    for subdir, files in WEB_ASSETS.items():
        target_subdir = os.path.join(target_dir, subdir)
        os.makedirs(target_subdir, exist_ok=True)
        for fname in files:
            source_path = os.path.join(source_dir, fname) # Look directly in source_dir
            # Also check within potential subdirs like 'templates' or 'static/js'
            if not os.path.exists(source_path):
                source_path = os.path.join(source_dir, subdir, fname)
            if not os.path.exists(source_path) and subdir == 'templates':
                 source_path = os.path.join(source_dir, 'templates', fname)

            if os.path.exists(source_path):
                 target_path = os.path.join(target_subdir, os.path.basename(fname))
                 try:
                     shutil.copy2(source_path, target_path)
                     logger.info(f"Copied web asset: {fname} to {subdir}/")
                     copied_count += 1
                 except Exception as e:
                     logger.warning(f"Could not copy web asset {fname}: {e}")
            else:
                 logger.debug(f"Web asset not found, skipping: {fname}")

    # Special handling for index/visualization HTML - choose one
    index_copied = False
    vis_template_copied = False
    target_template_dir = os.path.join(target_dir, "templates")
    for html_file in WEB_ASSETS.get("templates", []):
        source_path = os.path.join(source_dir, html_file)
        if not os.path.exists(source_path): source_path = os.path.join(source_dir,"templates",html_file) # Check template dir

        if os.path.exists(source_path):
            # Prefer quantum-kaleidoscope.html as base
            if "quantum-kaleidoscope.html" in html_file and not index_copied:
                shutil.copy2(source_path, os.path.join(target_template_dir, "index.html"))
                shutil.copy2(source_path, os.path.join(target_template_dir, "visualization.html"))
                logger.info(f"Used {html_file} for index.html and visualization.html")
                index_copied = True
                vis_template_copied = True
                break # Prioritize this one
            elif "advanced-cube.html" in html_file and not index_copied: # Fallback
                 shutil.copy2(source_path, os.path.join(target_template_dir, "index.html"))
                 logger.info(f"Used {html_file} for index.html (fallback)")
                 index_copied = True

    if not index_copied: logger.warning("Could not find a suitable HTML file for index.html")
    if not vis_template_copied: logger.warning("Could not find a suitable HTML file for visualization.html")

    logger.info(f"Copied {copied_count} web assets.")


# --- Main Integration Logic ---

def integrate_system(source_dir: str, target_dir: str):
    """Performs the integration process."""
    logger.info(f"Starting integration from '{source_dir}' to '{target_dir}'")

    # --- ADDED CHECK ---
    if not os.path.isdir(source_dir):
        logger.critical(f"Source directory does not exist or is not a directory: {source_dir}")
        return False
    # -------------------

    # 1. Create target directories
    try:
        create_directories(target_dir)
    except Exception as e:
        logger.critical(f"Failed to create target directories: {e}. Aborting.")
        return False

    # 2. Copy and rename core components
    integration_module_path = None
    core_system_path = None
    all_copied = True
    for target_name, sources in TARGET_FILES.items():
        source_path = find_best_source(sources, source_dir)
        if not source_path:
            logger.error(f"Missing critical component source for: {target_name}")
            all_copied = False
            continue # Skip if no source found

        if not copy_and_rename(source_path, target_name, target_dir):
            all_copied = False # Mark failure but continue trying others

        # Store paths for later use
        if target_name == "visualization_integration.py":
            integration_module_path = os.path.join(target_dir, target_name)
        if target_name == "unified_quantum_kaleidoscope.py":
             core_system_path = os.path.join(target_dir, target_name)

    if not all_copied:
         logger.error("Failed to copy one or more critical components. Integration incomplete.")
         # Decide if you want to stop or continue with potentially missing parts
         # return False # Uncomment to stop on critical copy failure

    # 3. Apply visualization patch
    if core_system_path and integration_module_path:
        if not apply_visualization_patch(core_system_path, integration_module_path):
            logger.warning("Failed to automatically apply visualization patch. Manual review might be needed.")
    else:
        logger.warning("Could not apply visualization patch: Core system or integration module path missing.")

    # 4. Copy essential supporting modules
    copy_essential_modules(source_dir, target_dir)

    # 5. Copy web assets
    copy_web_assets(source_dir, target_dir)

    logger.info("Integration process completed.")
    if not all_copied:
        logger.warning("Some critical files might be missing or failed to copy.")
    return True


# --- Command Line Execution ---

if __name__ == "__main__":
    import platform # Moved import here to be available globally if needed
    parser = argparse.ArgumentParser(description="Quantum Kaleidoscope Integration Script")
    parser.add_argument("source_dir", help="Directory containing the original script files.")
    parser.add_argument("target_dir", help="Directory where the integrated system will be created.")
    parser.add_argument("--force", action="store_true", help="Overwrite target directory if it exists.")

    args = parser.parse_args()

    source_directory = os.path.abspath(args.source_dir)
    target_directory = os.path.abspath(args.target_dir)

    # --- Check source directory existence *before* removing target ---
    if not os.path.isdir(source_directory):
        logger.critical(f"Source directory does not exist: {source_directory}")
        sys.exit(1)
    # ---------------------------------------------------------------

    if os.path.exists(target_directory):
        if args.force:
            logger.warning(f"Target directory {target_directory} exists. Overwriting (--force specified).")
            try:
                shutil.rmtree(target_directory)
            except Exception as e:
                 logger.critical(f"Failed to remove existing target directory: {e}")
                 sys.exit(1)
        else:
            logger.critical(f"Target directory {target_directory} already exists. Use --force to overwrite.")
            sys.exit(1)

    # Perform the integration
    if integrate_system(source_directory, target_directory):
        print("\nIntegration successful!")
        print(f"Integrated system created in: {target_directory}")
        print("\nNext Steps:")
        print(f"1. cd {target_directory}")
        print("2. (Optional) Create and activate a Python virtual environment:")
        print("   python3 -m venv venv")
        print("   source venv/bin/activate  # or venv\\Scripts\\activate.bat on Windows")
        print("3. Install dependencies (examine imports in .py files):")
        print("   pip install numpy websockets fastapi uvicorn flask requests ...")
        print("4. Run the system using the launcher:")
        print("   python quantum_kaleidoscope_launcher.py")
    else:
        print("\nIntegration failed. Please check the logs for errors.")
        sys.exit(1)
