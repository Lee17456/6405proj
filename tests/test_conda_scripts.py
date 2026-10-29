"""Exercise Bash environment routing using fake Conda/Python; no downloads or GPU."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASH = (r"C:\Program Files\Git\bin\bash.exe" if os.name == "nt" else shutil.which("bash"))


@unittest.skipUnless(BASH and Path(BASH).exists(), "Bash unavailable")
class CondaScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.project = self.base / "project with spaces"
        shutil.copytree(ROOT / "scripts", self.project / "scripts")
        shutil.copyfile(ROOT / "requirements.txt", self.project / "requirements.txt")
        shutil.copyfile(ROOT / "requirements-mamba.txt", self.project / "requirements-mamba.txt")
        self.conda = self.base / "fake conda"
        self.tools = self.base / "tools"
        self.log = self.base / "commands.log"
        self.write(self.tools / "uname", '#!/usr/bin/env bash\nif [[ "$1" == -s ]]; then echo Linux; else echo x86_64; fi\n')
        self.write(self.tools / "tmux", '''#!/usr/bin/env bash
if [[ "$1" == has-session ]]; then exit 1; fi
printf 'tmux' >> "$FAKE_LOG"; printf '|%s' "$@" >> "$FAKE_LOG"; printf '\\n' >> "$FAKE_LOG"
''')
        self.write(self.base / "fake-python", '''#!/usr/bin/env bash
printf 'python|%s' "${CONDA_PREFIX:-NONE}" >> "$FAKE_LOG"
printf '|%s' "$@" >> "$FAKE_LOG"; printf '\\n' >> "$FAKE_LOG"
''')
        self.write(self.conda / "bin/conda", '''#!/usr/bin/env bash
set -eu
printf 'conda' >> "$FAKE_LOG"; printf '|%s' "$@" >> "$FAKE_LOG"; printf '\\n' >> "$FAKE_LOG"
case "$1" in
  info) printf '%s\\n' "$FAKE_CONDA_BASE" ;;
  create)
    while [[ $# -gt 0 ]]; do
      if [[ "$1" == --prefix ]]; then prefix="$2"; shift; fi
      shift
    done
    mkdir -p "$prefix/bin" "$prefix/conda-meta"
    printf 'fixture\\n' > "$prefix/conda-meta/history"
    cp "$FAKE_PYTHON" "$prefix/bin/python"
    chmod +x "$prefix/bin/python"
    ;;
  list) printf '@EXPLICIT\\n' ;;
  *) exit 9 ;;
esac
''')
        self.write(self.conda / "etc/profile.d/conda.sh", '''conda() {
  [[ "$1" == activate ]] || return 8
  export CONDA_PREFIX="$2"
  export PATH="$2/bin:$PATH"
}
''')
        self.env = dict(os.environ)
        for key in ("CONDA_PREFIX", "CONDA_EXE", "PROJ6405_CONDA_EXE", "PYTHONPATH"):
            self.env.pop(key, None)
        self.env.update(FAKE_LOG=self.log.as_posix(), FAKE_CONDA_BASE=self.conda.as_posix(),
                        FAKE_PYTHON=(self.base / "fake-python").as_posix(),
                        PROJ6405_CONDA_EXE=(self.conda / "bin/conda").as_posix())

    def tearDown(self):
        self.temp.cleanup()

    def write(self, path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        path.chmod(0o755)

    def shell(self, text, *, fresh=False, success=True):
        env = dict(self.env)
        if fresh:
            env.pop("PROJ6405_CONDA_EXE", None)
        result = subprocess.run([BASH, "--noprofile", "--norc", "-c",
                                 'set -euo pipefail; cd "$1"; export PATH="$(cd "$2" && pwd):$PATH"; ' + text,
                                 "--", self.project.as_posix(), self.tools.as_posix()],
                                env=env, text=True, capture_output=True, timeout=30)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_setup_installs_only_into_project_and_preserves_old_venv(self):
        sentinel = self.project / ".venv/keep.txt"
        self.write(sentinel, "keep")
        self.shell("bash scripts/setup_conda.sh")
        log = self.log.read_text()
        self.assertIn("conda|create|--yes|--prefix|", log)
        self.assertIn("|--override-channels|--channel|conda-forge|python=3.11|pip", log)
        self.assertIn("|torch==2.5.1+cu121|--index-url|https://download.pytorch.org/whl/cu121", log)
        for line in log.splitlines():
            if line.startswith("python|"):
                self.assertTrue(line.split("|")[1].endswith("/project with spaces/.conda-env"), line)
        self.assertEqual(sentinel.read_text(), "keep")
        self.assertTrue((self.project / "reports/conda-explicit-linux64.txt").exists())

    def test_rerun_reuses_environment_and_legacy_entry_routes_to_conda(self):
        self.shell("bash scripts/setup_linux.sh")
        self.shell("bash scripts/setup_conda.sh", fresh=True)
        self.assertEqual(self.log.read_text().count("conda|create|"), 1)

    def test_existing_non_conda_directory_is_not_overwritten(self):
        sentinel = self.project / ".conda-env/keep.txt"
        self.write(sentinel, "keep")
        result = self.shell("bash scripts/setup_conda.sh", success=False)
        self.assertIn("not a completed Conda environment", result.stderr)
        self.assertEqual(sentinel.read_text(), "keep")
        self.assertFalse(self.log.exists())

    def test_activation_and_services_work_in_fresh_shell(self):
        self.shell("bash scripts/setup_conda.sh")
        self.log.write_text("")
        self.shell("source scripts/activate.sh; python --version", fresh=True)
        self.shell("bash scripts/install_mamba.sh", fresh=True)
        self.shell("bash scripts/run_api.sh", fresh=True)
        self.shell("UI_HOST=0.0.0.0 bash scripts/run_ui.sh", fresh=True)
        log = self.log.read_text()
        self.assertIn("|-m|uvicorn|proj6405.api:app|--host|127.0.0.1", log)
        self.assertIn("|-m|streamlit|run|app.py|--server.address|0.0.0.0", log)
        self.assertIn("|--no-build-isolation|-r|requirements-mamba.txt", log)
        self.assertNotIn("python|NONE", log)

    def test_tmux_receives_conda_path_and_ui_address_explicitly(self):
        self.shell("bash scripts/setup_conda.sh")
        self.log.write_text("")
        self.shell("RUN_PREFIX=full_ UI_HOST=0.0.0.0 bash scripts/start_services.sh", fresh=True)
        calls = [line for line in self.log.read_text().splitlines() if line.startswith("tmux|")]
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertIn("|-e|UI_HOST=0.0.0.0|-e|PROJ6405_CONDA_EXE=", call)
            self.assertIn("|-e|RUN_PREFIX=full_", call)

    def test_missing_environment_and_execute_instead_of_source_fail_clearly(self):
        missing = self.shell("source scripts/activate.sh", success=False)
        self.assertIn("environment missing", missing.stderr)
        wrong = self.shell("bash scripts/activate.sh", success=False)
        self.assertIn("source scripts/activate.sh", wrong.stderr)


if __name__ == "__main__":
    unittest.main()
