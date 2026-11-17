"""Freeze the instances of the registered RoboBallet / cuArms presets into the package, so that building a preset
needs no IK or convex decomposition (the 8-arm welding preset otherwise takes about 8 minutes).

    python scripts/freeze_roboballet_instances.py            # all presets without a frozen instance
    python scripts/freeze_roboballet_instances.py --rebuild  # regenerate all (after changing construction; also
                                                             # bump _CACHE_VERSION in roboballet_instances.py)

Each preset is built through get_env_by_name with an empty temporary cache, and the instance it writes there is moved
to assets/roboballet/instances. The file name is the hash of the construction options and the release checksum.
"""
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

cache = Path(tempfile.mkdtemp(prefix="mrmg-freeze-"))
os.environ["XDG_CACHE_HOME"] = str(cache)
os.environ.pop("MRMG_ROBOBALLET_ASSETS", None)  # freeze for the packaged assets only

from multi_robot_multi_goal_planning.problems import get_env_by_name  # noqa: E402
from multi_robot_multi_goal_planning.problems.core.registry import list_envs  # noqa: E402
from multi_robot_multi_goal_planning.problems.roboballet_instances import PACKAGED_ASSETS  # noqa: E402

target = PACKAGED_ASSETS / "instances"
if "--rebuild" in sys.argv:
    shutil.rmtree(target, ignore_errors=True)
target.mkdir(parents=True, exist_ok=True)
names = [n for n in list_envs() if n.startswith(("roboballet.", "cuarms."))]
for name in names:
    t = time.time()
    get_env_by_name(name)
    made = list((cache / "mrmg" / "roboballet" / "instances").glob("*.json"))
    for f in made:
        shutil.move(str(f), target / f.name)
    print(f"{name}: {'froze ' + ', '.join(f.name for f in made) if made else 'already frozen'} ({time.time() - t:.0f} s)",
          flush=True)
shutil.rmtree(cache, ignore_errors=True)
