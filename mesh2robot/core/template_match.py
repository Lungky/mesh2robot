"""Physics defaults for the URDF assembler — DB-lookup-free.

Earlier versions of this module retrieved per-joint limits, efforts,
velocities, and density from a 13-entry hand-curated `urdf_db.json`,
keyed on (DOF, joint-type sequence). For any input outside those 13
known robots, the lookup silently returned the closest stranger's
values — a pure matchmaking pattern that contradicts the project goal
("build a model not just a matchmaking tool", 2026-05-01 user
directive).

The lookup is gone. Joint limits now come from the **model's LimitsHead**
(trained on the 371 canonical robots' actual per-joint ranges) and are
optionally refined by the **PyBullet self-collision sweep**
(`mesh2robot.core.collision_sweep`).

What's left in this module is a small `Template` dataclass that the
assembler still consumes — but it now only carries non-learnable
physics defaults that no static-mesh signal can determine without
system identification: density, friction, damping, effort, velocity.
These get fixed conservative values; users can override per-robot via
the assembler's `final_limits_per_joint` Tier-1 input.

Future work: add mass / inertia / friction / damping heads to the
model so even those become learned, not hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass

import math


# Conservative physics defaults — fixed values, no DB lookup.
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
    score: float                            # kept for backwards compat; always 0
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


def make_default_template(query_dof: int, name: str = "<defaults>") -> Template:
    """Build a Template carrying only the non-learnable physics defaults.

    The assembler will combine this with model-predicted limits +
    collision-sweep refinement at URDF assembly time.
    """
    return Template(
        name=name,
        dof=query_dof,
        score=0.0,
        density=DEFAULT_DENSITY,
        friction=DEFAULT_FRICTION,
        damping=DEFAULT_DAMPING,
        effort_per_joint=[DEFAULT_EFFORT] * query_dof,
        velocity_per_joint=[DEFAULT_VELOCITY] * query_dof,
        limits_per_joint=[],
    )


def match(
    query_dof: int,
    query_types: list[str] | None = None,
    db: object | None = None,
) -> Template:
    """Backwards-compatibility shim — the old API used to do a DB lookup
    against `data/urdf_db.json` and return a "matched" template. That DB
    is gone; this now always returns the default-physics template.

    Callers should migrate to `make_default_template(query_dof)` directly.
    The `query_types` and `db` args are accepted for compatibility but
    ignored — they were only used for the lookup.
    """
    return make_default_template(query_dof)


if __name__ == "__main__":
    t = make_default_template(6)
    print("Defaults template (6 DOF):", t)
