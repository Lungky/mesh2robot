"""Fixed physics defaults for the URDF assembler.

This module USED TO be `template_match.py` and ran a 13-row lookup
against `data/urdf_db.json` to fetch density / friction / damping /
effort / velocity / limits keyed on (DOF, joint-type sequence). For
inputs outside the 13 known robots, it returned the closest stranger's
values — a matchmaking pattern that the project rejected (2026-05-02
user directive: "build a model not just a matchmaking tool. because
the custom robots will not be xarm6"). The DB and lookup are gone.

What remains:

  - `Template` dataclass — passed to the URDF assembler. Joint limits
    intentionally come from the **model's `LimitsHead`** + the optional
    PyBullet self-collision sweep at inference time, NOT from this
    module. The assembler reads them via `AssemblyInput.final_limits_per_joint`.
  - `make_default_template(dof)` — fills the dataclass with conservative
    fixed defaults. These are non-learnable physical properties no
    static-mesh signal can determine (you'd need system identification);
    treat them as starter values that the user can override per-robot.

Future work: add mass / inertia / friction / damping heads to the
model so even those become learned, not hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass

import math


# Conservative physics defaults — fixed values, no lookup.
# These are reasonable starting points for a "generic robot in metal":
#   - density 2700 kg/m³ matches aluminum (typical industrial-arm casing)
#   - friction 0.5 is a textbook Coulomb estimate for steel-on-steel
#   - damping 0.1 N·m·s/rad is a moderate joint damping
#   - effort 100 N·m is a mid-range joint torque
#   - velocity π rad/s ≈ 180°/s is industrial-arm typical
DEFAULT_DENSITY = 2700.0     # kg/m^3 (aluminum)
DEFAULT_FRICTION = 0.5
DEFAULT_DAMPING = 0.1
DEFAULT_EFFORT = 100.0       # N·m
DEFAULT_VELOCITY = math.pi   # rad/s


@dataclass
class Template:
    """Physics defaults bag. Consumed by the URDF assembler. Joint limits
    are intentionally empty — they come from the model's LimitsHead and
    the optional collision sweep, NOT from this module."""
    name: str
    dof: int
    density: float
    friction: float
    damping: float
    effort_per_joint: list[float]
    velocity_per_joint: list[float]
    # Always empty; assembler consults the model+sweep limits via
    # AssemblyInput.final_limits_per_joint (Tier 1) instead.
    limits_per_joint: list[tuple[float, float]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.limits_per_joint is None:
            self.limits_per_joint = []


def make_default_template(dof: int, name: str = "<defaults>") -> Template:
    """Build a Template carrying only the non-learnable physics defaults.

    The assembler will combine this with model-predicted limits +
    collision-sweep refinement at URDF assembly time.
    """
    return Template(
        name=name,
        dof=dof,
        density=DEFAULT_DENSITY,
        friction=DEFAULT_FRICTION,
        damping=DEFAULT_DAMPING,
        effort_per_joint=[DEFAULT_EFFORT] * dof,
        velocity_per_joint=[DEFAULT_VELOCITY] * dof,
        limits_per_joint=[],
    )


if __name__ == "__main__":
    t = make_default_template(6)
    print("Defaults template (6 DOF):", t)
