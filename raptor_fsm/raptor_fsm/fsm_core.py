# Copyright 2026 Aycan Coskun
# SPDX-License-Identifier: AGPL-3.0-only

"""
Six-state raptor avoidance state machine, independent of ROS.

Port of state_machine_v3.py with the transitions and parameter values of the paper
(icSoftComp 2026, Fig. 1). One call to RaptorFsm.step() is one tick of the original
4 Hz loop, and the FSM moves by at most one state per tick. Distances are in metres
and rates in m/s (the original used mm and mm/s). Velocities are NED (vx north,
vy east, vz down), as in the original set_position_target_local_ned commands.
"""

from dataclasses import dataclass, field
import math
from typing import Optional, Tuple

IDLE = 'IDLE'
OBSTACLE_AVOID = 'OBSTACLE_AVOID'
EARLY_WARNING = 'EARLY_WARNING'
THREAT_CONFIRMED = 'THREAT_CONFIRMED'
ESCAPE = 'ESCAPE'
RTL = 'RTL'

LAYER_NONE = 'NONE'
LAYER_OBSTACLE = 'OBSTACLE'
LAYER_RAPTOR = 'RAPTOR'

# Same values as the SECTOR_* constants in raptor_interfaces/msg/LidarHazard.
SECTOR_NONE = 0
SECTOR_FRONT = 1
SECTOR_FRONT_RIGHT = 2
SECTOR_RIGHT = 3
SECTOR_BACK = 4
SECTOR_LEFT = 5
SECTOR_FRONT_LEFT = 6

VelocityNed = Tuple[float, float, float]


@dataclass
class FsmParams:
    """Decision parameters; defaults are the values reported in the paper."""

    w_yolo: float = 0.6
    w_lidar: float = 0.4
    min_yolo_conf: float = 0.05       # fusion is 0 at or below this (LiDAR alone never fuses)
    d_warn_m: float = 3.0             # proximity score 0 at or beyond this distance
    d_hazard_m: float = 1.2           # proximity score 1 at or below this distance
    d_confirm_m: float = 1.5          # THREAT_CONFIRMED only below this distance
    fusion_warn: float = 0.35
    fusion_confirm: float = 0.65
    approach_rate_mps: float = 0.05   # 50 mm/s
    escape_duration_s: float = 5.0
    early_warn_timeout_s: float = 5.0
    escape_vx: float = 3.0
    escape_vy: float = 3.0
    escape_vz_up: float = 1.5
    escape_vz_down: float = 0.5
    avoid_speed: float = 1.5          # lateral obstacle avoidance speed
    avoid_back_speed: float = 1.0     # forward speed when the obstacle is behind
    avoid_full_speed_m: float = 2.0   # speed factor 1.0 beyond this distance
    avoid_stop_m: float = 1.0         # speed factor 0.0 at or below this distance


@dataclass
class FsmInputs:
    """Sensor snapshot for one tick (missing data means 'nothing detected')."""

    yolo_hazard: bool = False
    yolo_confirmed: bool = False
    yolo_conf: float = 0.0
    cx_norm: float = 0.5
    cy_norm: float = 0.5
    lidar_hazard: bool = False
    lidar_confirmed: bool = False
    distance_m: float = math.inf
    approach_rate_mps: float = 0.0
    sector: int = SECTOR_NONE


@dataclass
class FsmCommand:
    """What the tick asks the flight controller to do."""

    mode: Optional[str] = None                # flight mode to set, e.g. 'GUIDED' or 'RTL'
    velocity_ned: Optional[VelocityNed] = None
    label: str = ''


@dataclass
class TickResult:
    """Output of one tick, mirrors one row of state_log_v3.csv."""

    state: str
    previous_state: str
    layer_active: str
    fusion_score: float
    proximity_score: float
    command: FsmCommand = field(default_factory=FsmCommand)
    note: str = ''

    @property
    def changed(self) -> bool:
        return self.state != self.previous_state


def clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def proximity_score(distance_m: float, params: FsmParams) -> float:
    """P = clip((D_warn - d) / (D_warn - D_hazard), 0, 1); 0 when nothing is in range."""
    if not math.isfinite(distance_m):
        return 0.0
    return clip((params.d_warn_m - distance_m) / (params.d_warn_m - params.d_hazard_m))


def fusion_score(yolo_conf: float, proximity: float, params: FsmParams) -> float:
    """T = clip(w_y * C + w_l * P, 0, 1); 0 without a bird so LiDAR alone cannot fuse."""
    if yolo_conf <= params.min_yolo_conf:
        return 0.0
    return clip(params.w_yolo * yolo_conf + params.w_lidar * proximity)


def escape_velocity(cx_norm: float, cy_norm: float, params: FsmParams) -> VelocityNed:
    """Fly away from the bird: right if it is on the left of the frame, down if above."""
    vy = params.escape_vy if cx_norm < 0.5 else -params.escape_vy
    vz = params.escape_vz_down if cy_norm < 0.5 else -params.escape_vz_up
    return (params.escape_vx, vy, vz)


def obstacle_velocity(sector: int, distance_m: float, params: FsmParams) -> VelocityNed:
    """Sidestep away from the blocked sector, slower when the obstacle is closer."""
    speed = params.avoid_speed
    direction = {
        SECTOR_FRONT: (0.0, speed, 0.0),
        SECTOR_FRONT_LEFT: (0.0, speed, 0.0),
        SECTOR_LEFT: (0.0, speed, 0.0),
        SECTOR_RIGHT: (0.0, -speed, 0.0),
        SECTOR_FRONT_RIGHT: (0.0, -speed, 0.0),
        SECTOR_BACK: (params.avoid_back_speed, 0.0, 0.0),
    }.get(sector, (0.0, 0.0, 0.0))
    if distance_m > params.avoid_full_speed_m:
        factor = 1.0
    elif distance_m > params.avoid_stop_m:
        factor = 0.5
    else:
        # Too close: stop and hold (paper). OBSTACLE_AVOID is only left when the hazard
        # clears, so a vehicle held inside this distance stays there (see step 7 tests).
        factor = 0.0
    return tuple(factor * v for v in direction)


def layer_for(state: str) -> str:
    if state in (EARLY_WARNING, THREAT_CONFIRMED, ESCAPE, RTL):
        return LAYER_RAPTOR
    if state == OBSTACLE_AVOID:
        return LAYER_OBSTACLE
    return LAYER_NONE


class RaptorFsm:
    """IDLE, OBSTACLE_AVOID, EARLY_WARNING, THREAT_CONFIRMED, ESCAPE and RTL (terminal)."""

    def __init__(self, params: Optional[FsmParams] = None):
        self.params = params or FsmParams()
        self.state = IDLE
        self._early_warn_start = 0.0
        self._escape_start = 0.0
        self._escape_velocity: VelocityNed = (0.0, 0.0, 0.0)

    def step(self, inputs: FsmInputs, now_s: float) -> TickResult:
        """Run one tick at time now_s (seconds, any monotonic clock)."""
        p = self.params
        d = inputs.distance_m
        proximity = proximity_score(d, p)
        fusion = fusion_score(inputs.yolo_conf, proximity, p)
        previous = self.state
        command = FsmCommand()
        note = ''

        if self.state == IDLE:
            if fusion >= p.fusion_warn:
                self._enter_early_warning(now_s)
                note = f'FUSION_WARN fusion={fusion:.3f} d={d:.2f}m'
            elif inputs.lidar_hazard and d < p.d_warn_m:
                self.state = OBSTACLE_AVOID
                velocity = obstacle_velocity(inputs.sector, d, p)
                command = FsmCommand('GUIDED', velocity, 'GUIDED+OBS_AVOID')
                note = f'LIDAR_HAZARD sector={inputs.sector} d={d:.2f}m'

        elif self.state == OBSTACLE_AVOID:
            if fusion >= p.fusion_warn:
                self._enter_early_warning(now_s)
                note = f'RAPTOR_OVERRIDE fusion={fusion:.3f}'
            elif not inputs.lidar_hazard:
                self.state = IDLE
                command = FsmCommand(None, (0.0, 0.0, 0.0), 'HOVER')
                note = 'SECTOR_CLEAR'
            else:
                velocity = obstacle_velocity(inputs.sector, d, p)
                command = FsmCommand(None, velocity, 'OBS_AVOID')
                note = f'AVOIDING sector={inputs.sector} d={d:.2f}m'

        elif self.state == EARLY_WARNING:
            elapsed = now_s - self._early_warn_start
            if fusion >= p.fusion_confirm and d < p.d_confirm_m:
                self.state = THREAT_CONFIRMED
                note = f'FUSION_CONFIRM fusion={fusion:.3f} d={d:.2f}m'
            elif not inputs.yolo_hazard and elapsed > p.early_warn_timeout_s:
                self.state = IDLE
                note = f'TIMEOUT_CLEAR elapsed={elapsed:.1f}s'

        elif self.state == THREAT_CONFIRMED:
            if inputs.approach_rate_mps > p.approach_rate_mps:
                self.state = ESCAPE
                self._escape_start = now_s
                self._escape_velocity = escape_velocity(inputs.cx_norm, inputs.cy_norm, p)
                command = FsmCommand('GUIDED', self._escape_velocity, 'GUIDED+ESCAPE')
                note = (f'APPROACH_RATE={inputs.approach_rate_mps:.2f}m/s '
                        f'cx={inputs.cx_norm:.2f} cy={inputs.cy_norm:.2f}')
            elif d >= p.d_confirm_m or not inputs.yolo_confirmed:
                self._enter_early_warning(now_s)
                note = f'CONFIRM_LOST d={d:.2f}m yolo_confirmed={int(inputs.yolo_confirmed)}'

        elif self.state == ESCAPE:
            elapsed = now_s - self._escape_start
            if elapsed >= p.escape_duration_s:
                self.state = RTL
                command = FsmCommand('RTL', None, 'RTL')
                note = f'ESCAPE_ELAPSED={elapsed:.1f}s'
            else:
                # Fresh direction while the bird is visible; otherwise keep the last one
                # (v3 fell back to cx = cy = 0.5, which could turn the drone towards the bird).
                if inputs.yolo_hazard:
                    self._escape_velocity = escape_velocity(inputs.cx_norm, inputs.cy_norm, p)
                command = FsmCommand(None, self._escape_velocity, 'ESCAPE')
                note = f'ESCAPING elapsed={elapsed:.1f}s'

        elif self.state == RTL:
            note = 'HOLDING_RTL'

        return TickResult(
            state=self.state, previous_state=previous, layer_active=layer_for(self.state),
            fusion_score=fusion, proximity_score=proximity, command=command, note=note)

    def _enter_early_warning(self, now_s: float) -> None:
        self.state = EARLY_WARNING
        self._early_warn_start = now_s
