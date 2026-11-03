# Copyright 2026 Aycan Coskun
# SPDX-License-Identifier: AGPL-3.0-only

import math

import pytest

from raptor_fsm.fsm_core import (
    EARLY_WARNING, ESCAPE, escape_velocity, FsmInputs, FsmParams, fusion_score, IDLE,
    LAYER_NONE, LAYER_OBSTACLE, LAYER_RAPTOR, OBSTACLE_AVOID, obstacle_velocity,
    proximity_score, RaptorFsm, RTL, SECTOR_BACK, SECTOR_FRONT, SECTOR_FRONT_RIGHT,
    SECTOR_NONE, THREAT_CONFIRMED)

P = FsmParams()
TICK = 0.25  # 4 Hz


def bird(conf=0.8, d=math.inf, rate=0.0, cx=0.3, cy=0.6, confirmed=True):
    """Bird seen by the camera, closest LiDAR object at d."""
    return FsmInputs(
        yolo_hazard=True, yolo_confirmed=confirmed, yolo_conf=conf, cx_norm=cx, cy_norm=cy,
        lidar_hazard=d < P.d_hazard_m, lidar_confirmed=d < P.d_hazard_m, distance_m=d,
        approach_rate_mps=rate, sector=SECTOR_FRONT if d < P.d_hazard_m else SECTOR_NONE)


def obstacle(d=1.1, sector=SECTOR_FRONT):
    """Obstacle seen by the LiDAR, no bird."""
    return FsmInputs(lidar_hazard=True, lidar_confirmed=True, distance_m=d, sector=sector)


def run(fsm, inputs, start=0.0):
    """Feed one input per tick and return the results."""
    return [fsm.step(x, start + i * TICK) for i, x in enumerate(inputs)]


# Scores

def test_proximity_score_matches_paper_equation():
    assert proximity_score(3.0, P) == 0.0
    assert proximity_score(4.0, P) == 0.0
    assert proximity_score(1.2, P) == 1.0
    assert proximity_score(0.5, P) == 1.0
    assert proximity_score(2.1, P) == pytest.approx(0.5)
    assert proximity_score(math.inf, P) == 0.0


def test_fusion_score_paper_examples():
    # YOLO alone at full confidence reaches EARLY_WARNING but not THREAT_CONFIRMED.
    assert fusion_score(1.0, 0.0, P) == pytest.approx(0.6)
    assert fusion_score(0.85, 1.0, P) == pytest.approx(0.91)
    assert fusion_score(0.5, proximity_score(2.5, P), P) == pytest.approx(0.411, abs=1e-3)


def test_lidar_alone_cannot_fuse():
    assert fusion_score(0.0, 1.0, P) == 0.0
    assert fusion_score(0.05, 1.0, P) == 0.0
    assert fusion_score(0.06, 1.0, P) > 0.0


def test_escape_velocity_points_away_from_bird():
    assert escape_velocity(0.3, 0.6, P) == (3.0, 3.0, -1.5)    # left and low -> right and up
    assert escape_velocity(0.7, 0.4, P) == (3.0, -3.0, 0.5)    # paper Table 4: (3.0, -3.0, 0.5)


def test_obstacle_velocity_direction_and_speed_factor():
    assert obstacle_velocity(SECTOR_FRONT, 2.5, P) == (0.0, 1.5, 0.0)
    assert obstacle_velocity(SECTOR_FRONT_RIGHT, 1.5, P) == (0.0, -0.75, 0.0)
    assert obstacle_velocity(SECTOR_BACK, 1.5, P) == (0.5, 0.0, 0.0)
    assert obstacle_velocity(SECTOR_FRONT, 1.0, P) == (0.0, 0.0, 0.0)
    assert obstacle_velocity(SECTOR_NONE, 1.5, P) == (0.0, 0.0, 0.0)


# Transitions (paper Fig. 1)

def test_starts_idle_and_stays_idle_without_hazard():
    fsm = RaptorFsm()
    result = fsm.step(FsmInputs(), 0.0)
    assert result.state == IDLE
    assert result.layer_active == LAYER_NONE
    assert result.command.mode is None and result.command.velocity_ned is None


def test_full_raptor_pipeline_to_rtl():
    fsm = RaptorFsm()
    results = run(fsm, [
        bird(conf=0.7, d=2.0),             # fusion 0.64 -> EARLY_WARNING
        bird(conf=0.7, d=1.4),             # fusion 0.78, d < 1.5 -> THREAT_CONFIRMED
        bird(conf=0.7, d=1.3, rate=0.5),   # approaching -> ESCAPE
    ])
    assert [r.state for r in results] == [EARLY_WARNING, THREAT_CONFIRMED, ESCAPE]
    escape = results[-1]
    assert escape.layer_active == LAYER_RAPTOR
    assert escape.command.mode == 'GUIDED'
    assert escape.command.velocity_ned == (3.0, 3.0, -1.5)

    # Velocity is re-sent every tick for escape_duration_s, then RTL (terminal).
    start = 2 * TICK
    during = fsm.step(bird(d=1.3, rate=0.5), start + 4.75)
    assert during.state == ESCAPE and during.command.velocity_ned == (3.0, 3.0, -1.5)
    rtl = fsm.step(FsmInputs(), start + 5.0)
    assert rtl.state == RTL and rtl.command.mode == 'RTL'
    assert fsm.step(bird(d=0.5, rate=1.0), start + 6.0).state == RTL


def test_fsm_moves_at_most_one_state_per_tick():
    fsm = RaptorFsm()
    # Fusion is already above the confirm threshold and the bird is approaching.
    result = fsm.step(bird(conf=0.9, d=1.0, rate=1.0), 0.0)
    assert result.state == EARLY_WARNING


def test_early_warning_needs_close_range_for_threat_confirmed():
    fsm = RaptorFsm()
    run(fsm, [bird(conf=1.0, d=1.6)])
    # fusion = 0.6 + 0.4 * 0.78 = 0.91 >= 0.65, but d >= 1.5 m
    assert fsm.step(bird(conf=1.0, d=1.6), TICK).state == EARLY_WARNING
    assert fsm.step(bird(conf=1.0, d=1.45), 2 * TICK).state == THREAT_CONFIRMED


def test_yolo_alone_never_reaches_threat_confirmed():
    fsm = RaptorFsm()
    results = run(fsm, [bird(conf=1.0, d=math.inf)] * 10)
    assert {r.state for r in results} == {EARLY_WARNING}


def test_early_warning_times_out_only_without_bird():
    fsm = RaptorFsm()
    fsm.step(bird(conf=0.7, d=2.0), 0.0)
    assert fsm.step(FsmInputs(), 4.9).state == EARLY_WARNING     # before the 5 s timeout
    assert fsm.step(bird(conf=0.2), 5.5).state == EARLY_WARNING   # bird still visible
    result = fsm.step(FsmInputs(), 5.75)
    assert result.state == IDLE and result.note.startswith('TIMEOUT_CLEAR')


def test_threat_confirmed_returns_to_early_warning_when_bird_moves_away():
    fsm = RaptorFsm()
    run(fsm, [bird(conf=0.7, d=2.0), bird(conf=0.7, d=1.4)])
    assert fsm.state == THREAT_CONFIRMED
    assert fsm.step(bird(conf=0.7, d=1.6), 2 * TICK).state == EARLY_WARNING


def test_threat_confirmed_returns_to_early_warning_when_yolo_confirmation_lost():
    fsm = RaptorFsm()
    run(fsm, [bird(conf=0.7, d=2.0), bird(conf=0.7, d=1.4)])
    result = fsm.step(bird(conf=0.7, d=1.4, confirmed=False), 2 * TICK)
    assert result.state == EARLY_WARNING


def test_threat_confirmed_holds_while_bird_is_not_approaching():
    fsm = RaptorFsm()
    run(fsm, [bird(conf=0.7, d=2.0), bird(conf=0.7, d=1.4)])
    for i in range(2, 10):
        assert fsm.step(bird(conf=0.7, d=1.4, rate=0.04), i * TICK).state == THREAT_CONFIRMED


def test_obstacle_avoid_cycle():
    fsm = RaptorFsm()
    enter = fsm.step(obstacle(d=1.1, sector=SECTOR_FRONT), 0.0)
    assert enter.state == OBSTACLE_AVOID and enter.layer_active == LAYER_OBSTACLE
    assert enter.command.mode == 'GUIDED'
    assert enter.command.velocity_ned == (0.0, 0.75, 0.0)   # strafe right (east)
    keep = fsm.step(obstacle(d=1.1, sector=SECTOR_FRONT_RIGHT), TICK)
    assert keep.state == OBSTACLE_AVOID and keep.command.velocity_ned == (0.0, -0.75, 0.0)
    clear = fsm.step(FsmInputs(distance_m=1.5), 2 * TICK)
    assert clear.state == IDLE
    assert clear.command.velocity_ned == (0.0, 0.0, 0.0) and clear.command.label == 'HOVER'


def test_bird_overrides_obstacle_avoid():
    fsm = RaptorFsm()
    fsm.step(obstacle(d=1.1), 0.0)
    result = fsm.step(bird(conf=0.5, d=1.1), TICK)
    assert result.state == EARLY_WARNING and result.note.startswith('RAPTOR_OVERRIDE')


def test_bird_takes_priority_over_obstacle_from_idle():
    fsm = RaptorFsm()
    assert fsm.step(bird(conf=0.5, d=1.1), 0.0).state == EARLY_WARNING


def test_escape_keeps_last_direction_when_bird_is_lost():
    fsm = RaptorFsm()
    run(fsm, [bird(d=2.0, cx=0.2), bird(d=1.4, cx=0.2), bird(d=1.3, rate=0.5, cx=0.2)])
    assert fsm.state == ESCAPE
    lost = fsm.step(FsmInputs(), 3 * TICK)   # cx falls back to 0.5 in the inputs
    assert lost.command.velocity_ned == (3.0, 3.0, -1.5)
    moved = fsm.step(bird(d=1.3, cx=0.9, cy=0.1), 4 * TICK)
    assert moved.command.velocity_ned == (3.0, -3.0, 0.5)
