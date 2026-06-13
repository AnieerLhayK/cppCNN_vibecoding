# Agent Instructions

## Project Boundary

- This repository is the CNN project. Its Git root is the only default write scope.
- `codex/` is the C++ GTSRB implementation.
- `claude/` is the Python/PyTorch console car-logo prototype.
- Do not read from or write to `D:\AI\workspace` unless the user explicitly requests a cross-repository task.
- Do not treat agent names as permission to edit another implementation.
- Keep datasets, model weights, caches, and build outputs out of Git.

## Task Startup

1. Run `git status --short --branch`.
2. Confirm `git rev-parse --show-toplevel` resolves to this repository.
3. Work on a focused feature branch unless the user explicitly requests `main`.
4. Read the README for the implementation being changed.
5. Run that implementation's tests before committing.

## Validation

- Claude prototype: `python -m pytest -q claude/tests`
- Codex implementation: use the build and test commands in `codex/README.md`.
- Before handoff, report the active branch, commit, test result, and remaining work.
