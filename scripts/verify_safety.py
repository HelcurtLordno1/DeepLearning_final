"""CPU-only checks of the Windows lock and monitor; never initializes CUDA."""

import bootstrap  # Resolve src/ and the project root.

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import torch
import utils


def main():
    assert sys.platform == "win32"
    assert not torch.cuda.is_initialized()
    (utils.ROOT/"run/safety").mkdir(parents=True, exist_ok=True)
    previous = os.environ.pop("FINAL_PROJECT_MONITOR_PID", None)
    try:
        try:
            utils.windows_device()
            raise AssertionError("Unguarded GPU work was accepted")
        except RuntimeError as error:
            assert "monitored PowerShell" in str(error)
        assert not torch.cuda.is_initialized()

        config = utils.safe_training_config({"batch_size": 64, "workers": 2})
        assert config["batch_size"] == 16 and config["workers"] == 0

        with tempfile.TemporaryDirectory(dir=utils.ROOT/"run/safety") as temporary:
            folder = Path(temporary)
            path = folder/"best.pt"
            utils.save_checkpoint(path, {"weight": torch.tensor([1., 2.])})
            original_hash = utils.file_hash(path)
            normal_save = torch.save
            def interrupted_save(checkpoint, output):
                output.write(b"incomplete")
                raise OSError("simulated interrupted write")
            torch.save = interrupted_save
            try:
                try:
                    utils.save_checkpoint(path, {"weight": torch.tensor([3.])})
                    raise AssertionError("An interrupted write was not raised")
                except OSError:
                    pass
            finally:
                torch.save = normal_save
            assert utils.file_hash(path) == original_hash
            assert torch.equal(torch.load(path, weights_only=True)["weight"], torch.tensor([1., 2.]))
            utils.save_json(folder/"test.json", {"ok": True})
            utils.save_history(folder/"history.csv", [{"epoch": 1, "loss": .5}])
            assert (folder/"test.json").read_text(encoding="utf-8").startswith("{")
            assert "epoch,loss" in (folder/"history.csv").read_text(encoding="utf-8")

        utils.claim_training_slot()
        command = [sys.executable, "-c", "from utils import claim_training_slot; claim_training_slot()"]
        second = subprocess.run(command, cwd=utils.ROOT, capture_output=True, text=True, timeout=30)
        assert second.returncode != 0 and "already holds" in second.stderr
        utils._training_lock.close()
        utils._training_lock = None
        released = subprocess.run(command, cwd=utils.ROOT, capture_output=True, text=True, timeout=30)
        assert released.returncode == 0, released.stderr

        # A sleeping PowerShell process stands in for a monitor, without GPU work.
        monitor = subprocess.Popen(["powershell.exe", "-NoProfile", "-Command", "Start-Sleep -Seconds 20"])
        try:
            os.environ["FINAL_PROJECT_MONITOR_PID"] = str(monitor.pid)
            utils.connect_monitor()
            utils.check_monitor()
            monitor.terminate()
            monitor.wait(timeout=10)
            try:
                utils.check_monitor()
                raise AssertionError("A dead monitor was accepted")
            except RuntimeError as error:
                assert "monitor stopped" in str(error)
        finally:
            if monitor.poll() is None:
                monitor.terminate(); monitor.wait(timeout=10)

        for folder in ["src", "scripts", "task2/code", "task3/code"]:
            for path in (utils.ROOT/folder).glob("*.py"):
                compile(path.read_text(encoding="utf-8"), str(path), "exec")
        assert not torch.cuda.is_initialized()
        utils.save_json(utils.ROOT/"run/safety/python_guard_checks.json", {
            "platform": sys.platform, "executable": sys.executable,
            "unguarded_work_blocked": True, "concurrent_process_blocked": True,
            "lock_released_after_exit": True, "dead_monitor_detected": True,
            "reduced_config_recorded": True, "source_compiles": True,
            "interrupted_checkpoint_write_keeps_previous": True,
            "cuda_initialized": False, "training_started": False})
        print("Windows safety checks: PASS. CUDA was not initialized; no training started.")
    finally:
        if previous is None:
            os.environ.pop("FINAL_PROJECT_MONITOR_PID", None)
        else:
            os.environ["FINAL_PROJECT_MONITOR_PID"] = previous


if __name__ == "__main__":
    main()
