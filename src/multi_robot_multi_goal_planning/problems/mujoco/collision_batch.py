"""Optional official MuJoCo batches; no project-specific native extension."""

import numpy as np
import mujoco


class CollisionBatch:
    """Collision checks for many configurations on MuJoCo's native batch threads.

    Exact: a simulation without contacts is valid under every tolerance, so only its contact count is read from the
    batch. Configurations with contacts are re-checked by the environment's scalar check, in input order, which also
    leaves env.data as the scalar loop would at the first failure.
    """

    def __init__(self, model, size, workers):
        if type(size) is not int or type(workers) is not int or size < 1 or workers < 1:
            raise ValueError("batch_size and workers must be positive integers")
        try:
            from mujoco.batch import Batch
        except ImportError as exc:
            raise ImportError(
                "Official collision batching requires MuJoCo with mujoco.batch (tested with 3.17.0)"
            ) from exc
        self.model = model
        self.size = size
        self.batch = Batch(model, nsim=size, nthread=workers, persistent=True)
        self.state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self._ncon = self.batch.bind("ncon")
        self._qpos = self.batch.bind("qpos")
        self._base = np.empty(mujoco.mj_stateSize(model, self.state_spec))

    @staticmethod
    def _valid_rows(env, chunk):
        vectorized = getattr(env, "_configurations_are_valid", None)
        if vectorized is not None:
            return np.asarray(vectorized(chunk), dtype=bool)
        return np.asarray([env._configuration_is_valid(q) for q in chunk], dtype=bool)

    def check(self, env, rows, mode, tolerance):
        result = np.zeros(len(rows), dtype=bool)
        b = self.batch
        if not env.manipulating_env:
            # The scene state (mocap workpieces etc.) does not depend on the configuration: copy it once.
            mujoco.mj_getState(env.model, env.data, self._base, self.state_spec)
        for start in range(0, len(rows), self.size):
            chunk = rows[start : start + self.size]
            accepted = np.flatnonzero(self._valid_rows(env, chunk))
            if not len(accepted):
                continue
            ids = np.arange(len(accepted), dtype=np.int32)
            if env.manipulating_env:
                # Keep both free-joint and mocap attachment semantics in the
                # environment hook. Only collision work runs on native threads.
                for slot, i in enumerate(accepted):
                    env._prepare_collision_state(chunk[i], mode)
                    mujoco.mj_getState(
                        env.model, env.data, b.state[slot], self.state_spec
                    )
            else:
                b.state[ids] = self._base
                self._qpos[np.ix_(ids, env._all_robot_idx)] = chunk[accepted]
            b.forward(ids=ids, stage=mujoco.mjtStage.mjSTAGE_POS)
            ncon = self._ncon[ids]
            for slot, i in enumerate(accepted):
                # No contact: valid under any tolerance. Otherwise the scalar check decides (exact, and env.data
                # then holds that configuration's contacts).
                result[start + i] = bool(ncon[slot] == 0) or env.is_collision_free(chunk[i], mode, tolerance)
        return result
