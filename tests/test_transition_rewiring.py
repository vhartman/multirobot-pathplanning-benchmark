"""Shared transition roots retain the mode of their incoming motion edge."""
from types import SimpleNamespace
from dataclasses import dataclass

import numpy as np
import pytest

from multi_robot_multi_goal_planning.planners.rrtstar_base import BaseRRTstar


@dataclass(eq=False)
class ModeStub:
    name: str
    prev_mode: object = None


@pytest.mark.parametrize('new_mode,child_mode,free_mode,expected',[
    ('incoming','incoming','outgoing',False),
    ('incoming','incoming','incoming',True),
    ('incoming','outgoing','outgoing',True),
    ('outgoing','outgoing','outgoing',True),
    ('outgoing','incoming','incoming',False),
])
def test_rewiring_checks_motion_mode_not_tree_container(new_mode,child_mode,free_mode,expected):
    incoming=ModeStub('incoming')
    outgoing=ModeStub('outgoing',incoming)
    modes=dict(incoming=incoming,outgoing=outgoing)
    parent=SimpleNamespace(children=[])
    new=SimpleNamespace(id=1,state=SimpleNamespace(q='new',mode=modes[new_mode]),
                        cost=1.,parent=None,children=[])
    child=SimpleNamespace(id=2,state=SimpleNamespace(q='child',mode=modes[child_mode]),
                          cost=10.,cost_to_parent=10.,parent=parent,children=[object()])
    parent.children.append(child)
    planner=SimpleNamespace(trees={outgoing:SimpleNamespace(get_node=lambda *_:child)})
    checked=[]
    def check(a,b,mode):
        checked.append(mode)
        return mode is modes[free_mode]
    planner._check_edge_cf=check
    changed=BaseRRTstar.rewire(planner,outgoing,np.array([2]),new,np.array([2.]),np.array([10.]))
    assert changed is expected
    assert child.parent is (new if expected else parent)
    if checked:
        assert checked==[modes[child_mode]]
    if expected:
        assert child.cost==3. and child.cost_to_parent==2.
        assert child in new.children and child not in parent.children
