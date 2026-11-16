"""Optional official MuJoCo batches; no project-specific native extension."""

import numpy as np
import mujoco


class CollisionBatch:
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
        # Contacts cannot be bound as an array: get_data needs persistent data.
        self.batch = Batch(model, nsim=size, nthread=workers, persistent=True)
        self.state_spec = mujoco.mjtState.mjSTATE_INTEGRATION

    def check(self, env, rows, mode, tolerance):
        result = np.zeros(len(rows), dtype=bool)
        b = self.batch
        for start in range(0, len(rows), self.size):
            chunk = rows[start : start + self.size]
            accepted = np.asarray(
                [i for i, q in enumerate(chunk) if env._configuration_is_valid(q)],
                dtype=int,
            )
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
                mujoco.mj_getState(env.model, env.data, b.state[0], self.state_spec)
                b.state[ids] = b.state[0]
                b.qpos[np.ix_(ids, env._all_robot_idx)] = chunk[accepted]
            b.forward(ids=ids, stage=mujoco.mjtStage.mjSTAGE_POS)
            for slot, i in enumerate(accepted):
                b.get_data(slot, env.data)
                result[start + i] = env._contacts_are_valid(tolerance)
        return result
