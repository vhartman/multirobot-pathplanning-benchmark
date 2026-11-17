# RoboBallet assets (third-party)

This directory is a subset of the RoboBallet code and data release, redistributed unmodified. MRMG's `roboballet.*`
and `cuarms.*` environments use it.

- **Source:** "RoboBallet" release, Zenodo record 16546403, https://zenodo.org/records/16546403. The archive is
  `roboballet.tar.zst` (1.1 GB, MD5 `659f63c725f2237b3ccd10a9ba8f5717`). The code is at
  https://github.com/paichichi/roboballet.
- **Copyright** 2025 Google LLC.
- **Licenses.** As stated in the release README:
  - all software is licensed under the Apache License, Version 2.0 (`LICENSE` in this directory, copied from the
    release);
  - all other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY),
    https://creativecommons.org/licenses/by/4.0/legalcode.

  Both are distributed "AS IS", without warranties or conditions of any kind. This is not an official Google
  product.
- **What is included.** `data/mujoco_world` (MuJoCo world files and meshes) and `data/paper_test_tasksets` (the
  test targets), unchanged; nothing else from the release. The Panda model appears to derive from MuJoCo
  Menagerie's `franka_emika_panda` (Apache-2.0); the release ships it without a separate notice.
- **Changes.** None to the files here. MRMG builds its scenes from them at runtime, for example by lifting the
  workpiece for the 8-arm world as RoboBallet's evaluator does.
- **Not part of the release.** `instances/` holds scene instances that MRMG generated from these assets (IK goals,
  workpiece decomposition), written by `scripts/freeze_roboballet_instances.py`.
