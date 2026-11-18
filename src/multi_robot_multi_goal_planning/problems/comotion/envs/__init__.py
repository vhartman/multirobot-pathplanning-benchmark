"""Register native and serialized CoMotion environments on demand."""
from . import panda
from ...core.registry import register
from ..converted_env import load_comotion_problem

register("comotion.converted")(load_comotion_problem)
