"""Export supported RAI primitives and joints; RAI is used only during export."""
from multi_robot_multi_goal_planning.problems.planning_env import (
    SequenceMixin,
    DependencyGraphMixin,
)
from multi_robot_multi_goal_planning.problems.comotion.converted_env import (
    ConvertedSequenceEnvironment,
    ConvertedDependencyEnvironment,
)
import numpy as np

_JOINTS = {
    **{f'hinge{a}': [(0, i)] for i, a in enumerate('XYZ')},
    **{f'trans{a}': [(1, i)] for i, a in enumerate('XYZ')},
    'transXY': [(1, 0), (1, 1)],
    'trans3': [(1, 0), (1, 1), (1, 2)],
    'transXYPhi': [(1, 0), (1, 1), (0, 2)],
    'phiTransXY': [(0, 2), (1, 0), (1, 1)],
}


def export_rai_scene(source):
    import robotic as ry
    C = ry.Config()
    C.addConfigurationCopy(source.C_base)
    names = list(C.getJointNames())
    for robot in source.robots:
        coordinates = [name for joint in source.robot_joints[robot]
                       for name in names if name == joint or name.startswith(joint + ':')]
        if [names[int(i)] for i in source.robot_idx[robot]] != coordinates:
            raise NotImplementedError('Configuration and RAI joint orders differ')
    C.setJointState(np.zeros(len(names)))
    frames = C.getFrames()
    frame_names = [f.name for f in frames]
    index = {name: i for i, name in enumerate(frame_names)}
    parents, poses, has_joint, contacts = [], [], [], []
    joints = [None] * len(names)
    shape_frames, kinds, sizes = [], [], []
    movable = sorted({t.frames[1] for t in source.tasks
                      if t.type not in (None, 'goto') and t.frames and len(t.frames) > 1})
    for f in frames:
        info = f.info()
        parent = f.getParent()
        parents.append(-1 if parent is None else index[parent.name])
        poses.append([*f.getPosition(), *f.getQuaternion()] if parent is None
                     else [*f.getRelativePosition(), *f.getRelativeQuaternion()])
        has_joint.append(info.get('joint') is not None)
        contacts.append(int(info.get('contact', 0)))
        kind = info.get('joint')
        if f.name in movable and kind not in (None, 'rigid'):
            raise NotImplementedError('Manipulated roots must be rigid objects')
        if kind not in (None, 'rigid'):
            if kind not in _JOINTS or info.get('mimic') or info.get('scale', 1.) != 1.:
                raise NotImplementedError(f'{f.name}: unsupported joint {info}')
            axes = _JOINTS[kind]
            for k, (typ, axis) in enumerate(axes):
                name = f.name if len(axes) == 1 else f'{f.name}:{k}'
                if name not in names:
                    raise NotImplementedError(f'Inactive or coupled joint {name}')
                joints[names.index(name)] = [index[f.name], typ, *np.eye(3)[axis]]
        if not contacts[-1] and f.name not in movable:
            continue
        shape = info.get('shape')
        size = np.asarray(f.getSize())
        if shape in (None, 'none', 'marker'):
            continue
        if shape == 'sphere':
            typ, dims = 0, [size[-1], 0., 0.]
        elif shape in ('capsule', 'cylinder'):
            typ, dims = (1 if shape == 'capsule' else 3), [size[-1], size[-2] / 2, 0.]
        elif shape == 'box' or (shape == 'ssBox' and size[-1] == 0):
            typ, dims = 2, (size[:3] / 2).tolist()
        else:
            raise NotImplementedError(f'{f.name}: unsupported shape {shape}; convert rounded boxes explicitly first')
        shape_frames.append(index[f.name]); kinds.append(typ); sizes.append(dims)
    if any(j is None for j in joints):
        raise NotImplementedError('Not all active coordinates could be exported')
    definition = dict(frame_names=frame_names, joint_names=names, parents=parents,
                      poses=poses, joints=joints, shape_frames=shape_frames, kinds=kinds, sizes=sizes,
                      limits=np.asarray(source.limits).T.tolist(), movable=movable,
                      robot_names=list(source.robots),
                      frame_robot={j.split(':')[0]: i for i, r in enumerate(source.robots)
                                   for j in source.robot_joints[r]},
                      start_q=source.start_pos.state().tolist(), tolerance=float(source.collision_tolerance),
                      has_joint=has_joint, contacts=contacts,
                      actions=[dict(type=t.type, frames=list(t.frames or []),
                                    indices=np.concatenate([source.robot_idx[r] for r in t.robots]).tolist(),
                                    side_effect=t.side_effect,
                                    side_effect_data=(None if t.side_effect_data is None else np.asarray(t.side_effect_data).tolist()))
                               for t in source.tasks])
    shape_index = {frame_names[f]: i for i, f in enumerate(shape_frames)}
    excluded = []
    for pair in getattr(source, 'collision_exclusions', ()):
        if len(pair) != 2 or not set(pair).issubset(shape_index):
            raise ValueError(f'Unknown collision shape in excluded pair: {pair}')
        excluded.append(sorted(shape_index[n] for n in pair))
    definition['excluded_pairs'] = sorted(excluded)
    return definition


def to_comotion(source):
    """Snapshot a supported geometric RAI problem as an independent CoMotion problem.

    Spheres, capsules, cylinders and sharp boxes retain their geometry. Rounded
    boxes, meshes, constraints and dynamic models require explicit conversion.
    Nonzero collision tolerances use the native solver's penetration depths,
    which can differ from RAI near the tolerance boundary.
    """
    if isinstance(source, SequenceMixin):
        return ConvertedSequenceEnvironment(source)
    if isinstance(source, DependencyGraphMixin):
        return ConvertedDependencyEnvironment(source)
    raise NotImplementedError('Only sequence/dependency-graph mode logic is supported')
