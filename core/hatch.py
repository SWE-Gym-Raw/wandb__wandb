"""Builds wandb-core."""

import os
import pathlib
import subprocess
from typing import Mapping, Optional


def build_wandb_core(
    go_binary: pathlib.Path,
    output_path: pathlib.PurePath,
    with_code_coverage: bool,
    with_race_detection: bool,
    wandb_commit_sha: Optional[str],
    target_system,
    target_arch,
) -> None:
    """Builds the wandb-core Go module.

    Args:
        go_binary: Path to the Go binary, which must exist.
        output_path: The path where to output the binary, relative to the
            workspace root.
        with_code_coverage: Whether to build the binary with code coverage
            support, using `go build -cover`.
        with_race_detection: Whether to build the binary with race detection
            enabled, using `go build -race`.
        wandb_commit_sha: The Git commit hash we're building from, if this
            is the https://github.com/wandb/wandb repository. Otherwise, an
            empty string.
        target_system: The target operating system (GOOS) or an empty string
            to use the current OS.
        target_arch: The target architecture (GOARCH) or an empty string
            to use the current architecture.
    """
    coverage_flags = ["-cover"] if with_code_coverage else []
    race_detect_flags = ["-race"] if with_race_detection else []
    output_flags = ["-o", str(".." / output_path)]

    ld_flags = _go_linker_flags(
        wandb_commit_sha=wandb_commit_sha,
        target_system=target_system,
        target_arch=target_arch,
    )
    ld_flags = [f"-ldflags={ld_flags}"]

    vendor_flags = ["-mod=vendor"]

    debug_flags = ["-gcflags=all=-N -l"]

    # We have to invoke Go from the directory with go.mod, hence the
    # paths relative to ./core
    subprocess.check_call(
        [
            str(go_binary),
            "build",
            *debug_flags,
            *coverage_flags,
            *race_detect_flags,
            *ld_flags,
            *output_flags,
            *vendor_flags,
            str(pathlib.Path("cmd", "wandb-core", "main.go")),
        ],
        cwd="./core",
        env=_go_env(
            with_race_detection=with_race_detection,
            target_system=target_system,
            target_arch=target_arch,
        ),
    )


def _go_linker_flags(
    wandb_commit_sha: Optional[str],
    target_system: str,
    target_arch: str,
) -> str:
    """Returns linker flags for the Go binary as a string."""
    flags = [
        # "-s",  # Omit the symbol table and debug info.
        # "-w",  # Omit the DWARF symbol table.
        # Set the Git commit variable in the main package.
        "-X",
        f"main.commit={wandb_commit_sha or 'unknown'}",
    ]

    if target_system == "linux" and target_arch == "amd64":
        ext_ld_flags = " ".join(
            [
                # Use https://en.wikipedia.org/wiki/Gold_(linker)
                "-fuse-ld=gold",
                # Set the --weak-unresolved-symbols option in gold, converting
                # unresolved symbols to weak references. This is necessary to
                # build a Go binary with cgo on Linux, where the NVML libraries
                # needed for Nvidia GPU monitoring may not be available at build time.
                "-Wl,--weak-unresolved-symbols",
            ]
        )
        flags += ["-extldflags", f'"{ext_ld_flags}"']

    return " ".join(flags)


def _go_env(
    with_race_detection: bool,
    target_system: str,
    target_arch: str,
) -> Mapping[str, str]:
    env = os.environ.copy()

    env["GOOS"] = target_system
    env["GOARCH"] = target_arch

    if with_race_detection:
        # Crash if a race is detected. The default behavior is to print
        # to stderr and continue.
        env["GORACE"] = "halt_on_error=1"

    if (target_system, target_arch) in [
        # Use cgo on AMD64 Linux to build dependencies needed for GPU metrics.
        ("linux", "amd64"),
        # Use cgo on ARM64 macOS for the gopsutil dependency, otherwise several
        # system metrics are unavailable.
        ("darwin", "arm64"),
    ] or (
        # On Windows, -race requires cgo.
        target_system == "windows" and with_race_detection
    ):
        env["CGO_ENABLED"] = "1"

    return env
