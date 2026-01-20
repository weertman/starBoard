"""
Python wrapper for Wildlife ReID Police Lineup tools
Provides easy programmatic access without command line usage
Fixed version with proper UTF-8 encoding for Windows
"""
# CRITICAL: Disable PyTorch Dynamo to avoid typing.Self issues with Python <3.11
import os
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['PYTORCH_DISABLE_LIBRARY_INIT'] = '1'

# Now continue with regular imports
import sys
import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

class LineupTools:
    """Wrapper class for Wildlife ReID lineup tools"""

    def __init__(self, reid_checkpoint: str, yolo_checkpoint: str):
        """
        Initialize with model paths

        Args:
            reid_checkpoint: Path to ReID model checkpoint
            yolo_checkpoint: Path to YOLO model checkpoint
        """
        self.reid_checkpoint = Path(reid_checkpoint)
        self.yolo_checkpoint = Path(yolo_checkpoint)

        # Validate checkpoints exist
        if not self.reid_checkpoint.exists():
            raise FileNotFoundError(f"ReID checkpoint not found: {reid_checkpoint}")
        if not self.yolo_checkpoint.exists():
            raise FileNotFoundError(f"YOLO checkpoint not found: {yolo_checkpoint}")

        # Store paths to scripts (all in same directory now)
        self.script_dir = Path(__file__).parent
        self.database_builder_script = self.script_dir / "database_builder.py"
        self.lineup_app_script = self.script_dir / "lineup_app.py"

    def build_database(self,
                       data_dir: str,
                       output_path: Optional[str] = None,
                       batch_size: int = 32,
                       single_outing: bool = False,
                       include_outing_prefix: bool = True,
                       run_async: bool = False) -> Dict[str, Any]:
        """
        Build embedding database from field data

        Args:
            data_dir: Directory containing field data
            output_path: Output database path (auto-generated if None)
            batch_size: Batch size for processing
            single_outing: If True, data_dir contains single outing
            include_outing_prefix: If True, include outing in identity
            run_async: If True, run in background and return immediately

        Returns:
            Dictionary with status and output information
        """
        # Auto-generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"lineup_database_{timestamp}.npz"

        # Build command
        cmd = [
            sys.executable,
            str(self.database_builder_script),
            "--reid-checkpoint", str(self.reid_checkpoint),
            "--yolo-checkpoint", str(self.yolo_checkpoint),
            "--data-dir", str(data_dir),
            "--output", str(output_path),
            "--batch-size", str(batch_size)
        ]

        if single_outing:
            cmd.append("--single-outing")

        if not include_outing_prefix:
            cmd.append("--no-outing-prefix")

        # Run command
        if run_async:
            # Run in background with UTF-8 encoding
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',  # UTF-8 encoding for Windows compatibility
                errors='replace'   # Replace any problematic characters
            )

            return {
                "status": "started",
                "process": process,
                "output_path": output_path,
                "command": " ".join(cmd)
            }
        else:
            # Run and wait for completion with UTF-8 encoding
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',  # UTF-8 encoding for Windows compatibility
                errors='replace'   # Replace any problematic characters
            )

            return {
                "status": "completed" if result.returncode == 0 else "failed",
                "return_code": result.returncode,
                "output_path": output_path,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd)
            }

    def launch_lineup_app(self,
                          database_path: str,
                          block: bool = True) -> subprocess.Popen:
        """
        Launch the lineup GUI application

        Args:
            database_path: Path to embedding database
            block: If True, wait for app to close; if False, return immediately

        Returns:
            Subprocess object
        """
        # Validate database exists
        if not Path(database_path).exists():
            raise FileNotFoundError(f"Database not found: {database_path}")

        # Build command
        cmd = [
            sys.executable,
            str(self.lineup_app_script),
            "--reid-checkpoint", str(self.reid_checkpoint),
            "--yolo-checkpoint", str(self.yolo_checkpoint),
            "--database", str(database_path)
        ]

        # Launch application
        if block:
            # Run and wait with UTF-8 encoding
            process = subprocess.run(
                cmd,
                encoding='utf-8',  # UTF-8 encoding for Windows compatibility
                errors='replace'   # Replace any problematic characters
            )
            return process
        else:
            # Run in background with UTF-8 encoding
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8',  # UTF-8 encoding for Windows compatibility
                errors='replace'   # Replace any problematic characters
            )
            return process

    def quick_build_and_launch(self,
                               data_dir: str,
                               single_outing: bool = False) -> None:
        """
        Convenience method to build database and immediately launch app

        Args:
            data_dir: Directory containing field data
            single_outing: If True, data_dir contains single outing
        """
        print("Building embedding database...")
        result = self.build_database(
            data_dir=data_dir,
            single_outing=single_outing,
            run_async=False
        )

        if result["status"] == "completed":
            print(f"Database built successfully: {result['output_path']}")

            # Verify file exists
            if Path(result['output_path']).exists():
                file_size = os.path.getsize(result['output_path']) / 1e6
                print(f"Database file size: {file_size:.2f} MB")
                print("\nLaunching lineup application...")

                self.launch_lineup_app(
                    database_path=result["output_path"],
                    block=True
                )
            else:
                print("Error: Database file was not created!")
                print("Check the output for errors:")
                print(result.get('stderr', 'No error output'))
        else:
            print("Database build failed!")
            print("STDERR:", result.get("stderr", "No error output"))
            if result.get("stdout"):
                print("STDOUT:", result["stdout"])


# Convenience functions for direct use

def build_database(reid_checkpoint: str,
                   yolo_checkpoint: str,
                   data_dir: str,
                   **kwargs) -> Dict[str, Any]:
    """
    Quick function to build database

    Example:
        result = build_database(
            "models/reid.pth",
            "models/yolo.pt",
            "../archive/sequence_sorted_r1"
        )
    """
    tools = LineupTools(reid_checkpoint, yolo_checkpoint)
    return tools.build_database(data_dir, **kwargs)


def launch_app(reid_checkpoint: str,
               yolo_checkpoint: str,
               database_path: str) -> None:
    """
    Quick function to launch lineup app

    Example:
        launch_app(
            "models/reid.pth",
            "models/yolo.pt",
            "lineup_database.npz"
        )
    """
    tools = LineupTools(reid_checkpoint, yolo_checkpoint)
    tools.launch_lineup_app(database_path, block=True)