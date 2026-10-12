"""Ball--Berry 与 Jarvis 气孔方案测试。"""

from noahmp_clm import ColumnState, Forcing, PhysicsOptions
from noahmp_clm.physics.canopy import ball_berry_rs, jarvis_rs, stomatal_resistance


def _forcing(swdown: float = 600.0, qair: float = 0.008) -> Forcing:
    return Forcing(1800.0, 288.0, qair, 2.0, 101325.0, swdown, 370.0, 0.0)


def test_ball_berry_is_reasonable_under_wet_canopy() -> None:
    state = ColumnState(options=PhysicsOptions(OptStomataResistance=1))
    f = _forcing()
    bb = ball_berry_rs(state, f)
    jarvis = jarvis_rs(state, f)
    assert state.rsmin <= bb <= 5000.0
    assert state.rsmin <= jarvis <= 5000.0
    assert max(bb, jarvis) / min(bb, jarvis) < 10.0


def test_stomatal_dispatch_changes_with_option() -> None:
    state = ColumnState()
    f = _forcing()
    assert stomatal_resistance(state, f, 1) == ball_berry_rs(state, f)
    assert stomatal_resistance(state, f, 2) == jarvis_rs(state, f)

