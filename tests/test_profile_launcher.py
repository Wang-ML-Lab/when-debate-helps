from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_launcher_stops_before_merge_when_a_worker_fails(tmp_path: Path) -> None:
    binary_dir = tmp_path / "bin"
    output_dir = tmp_path / "output"
    binary_dir.mkdir()
    output_dir.mkdir()
    fake_wdh = binary_dir / "wdh"
    fake_wdh.write_text(
        """#!/usr/bin/env bash
if [[ "$1" == "profile" ]]; then
  while [[ "$#" -gt 0 ]]; do
    if [[ "$1" == "--shard-id" ]]; then
      shift
      [[ "$1" == "0" ]] && exit 17
      exit 0
    fi
    shift
  done
fi
if [[ "$1" == "merge-profiles" ]]; then
  touch "${MERGE_MARKER}"
  exit 0
fi
""",
        encoding="utf-8",
    )
    fake_wdh.chmod(0o755)
    merge_marker = tmp_path / "merge-called"
    environment = dict(os.environ)
    environment["PATH"] = f"{binary_dir}:{environment['PATH']}"
    environment["MERGE_MARKER"] = str(merge_marker)

    result = subprocess.run(
        ["bash", "scripts/launch_profile_shards.sh", "config", "seeds", str(output_dir), "0", "1"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "profile shard 0 failed" in result.stderr
    assert not merge_marker.exists()
