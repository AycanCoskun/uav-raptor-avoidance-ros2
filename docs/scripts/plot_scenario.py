#!/usr/bin/env python3
# Copyright 2026 Aycan Coskun
# SPDX-License-Identifier: AGPL-3.0-only

"""
Plot a raptor scenario run exported with `ros2 run raptor_fsm export_run_csv`.

An event lane and four panels on a shared time axis (t_rel: t = 0 at the first non-zero
raptor rig command). The event lane marks the bird's detection, the state machine's
reaction and the time the bird would reach the drone without an escape, with the
intervals between them. The panels show the LiDAR distance with the paper's distance
thresholds and the bird's distance to the drone (also had the drone not escaped), YOLO
confidence and fusion score with the FSM thresholds, the drone's velocity, and the state
timeline. Times, speeds and the drone position come from the CSV; the rig's start
position comes from the scenario parameters and the rig model. Writes one PNG.
Usage: plot_scenario.py <run.csv> <output.png>
"""

import csv
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SCENARIO_PARAMS = REPO / 'raptor_bringup' / 'config' / 'raptor_params.yaml'
RIG_MODEL = REPO / 'raptor_bringup' / 'models' / 'raptor_rig' / 'model.sdf'

THEME = {
    'surface': '#fcfcfb', 'ink': '#0b0b0b', 'ink2': '#52514e', 'muted': '#898781',
    'grid': '#e1e0d9', 'axis': '#c3c2b7',
    'series': ['#2a78d6', '#eb6834', '#1baf7a', '#eda100'],
}
# State colours of the demo GIF (STATE_COLOURS in raptor_fsm/demo_recorder.py), RGB.
STATE_COLOURS = {
    'IDLE': (110, 110, 110),
    'OBSTACLE_AVOID': (40, 120, 220),
    'EARLY_WARNING': (230, 170, 0),
    'THREAT_CONFIRMED': (235, 100, 20),
    'ESCAPE': (210, 30, 30),
    'RTL': (30, 140, 70),
}
SHORT = {'IDLE': 'IDLE', 'EARLY_WARNING': 'EARLY_WARN.', 'THREAT_CONFIRMED': 'THREAT_CONF.',
         'ESCAPE': 'ESCAPE', 'RTL': 'RTL', 'OBSTACLE_AVOID': 'OBSTACLE_AVOID'}
EVENT_DASH = (0, (3, 2))


def number(text):
    try:
        value = float(text)
    except ValueError:
        return math.nan
    return value if math.isfinite(value) else math.nan


def load(path):
    rows = list(csv.DictReader(open(path)))
    if not rows or math.isnan(number(rows[0].get('t_rel', ''))):
        raise SystemExit(f'{path} has no t_rel: export the run again with export_run_csv')
    rtl = next(number(r['t_rel']) for r in rows if r['state'] == 'RTL')
    window = [r for r in rows if -1.0 <= number(r['t_rel']) <= rtl + 3.0]
    t = [number(r['t_rel']) for r in window]
    return t, window


def bird_start():
    """Bird picture centre east of the drone when the rig starts.

    The scenario places the pole start_distance_m east of the drone, and the picture is
    the pole's offset in the rig model ahead of it (scenario parameter bird_offset_m).
    """
    params = yaml.safe_load(SCENARIO_PARAMS.read_text())['scenario_node']['ros__parameters']
    pole = ET.parse(RIG_MODEL).find(".//visual[@name='pole']")
    return params['start_distance_m'] - float(pole.find('pose').text.split()[0])


def segments(t, rows):
    out = []
    for ti, r in zip(t, rows):
        if out and out[-1][0] == r['state']:
            out[-1][2] = ti
        else:
            out.append([r['state'], ti, ti])
    for i in range(len(out) - 1):
        out[i][2] = out[i + 1][1]
    return out


def luminance(rgb):
    """WCAG relative luminance of an RGB colour (0-255 per channel)."""
    lin = [c / 255 / 12.92 if c / 255 <= 0.04045 else ((c / 255 + 0.055) / 1.055) ** 2.4
           for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def state_fill(state, theme):
    """Band colour of a state and the text colour with the higher contrast on it."""
    rgb = STATE_COLOURS.get(state, STATE_COLOURS['IDLE'])
    ink = tuple(int(theme['ink'][i:i + 2], 16) for i in (1, 3, 5))
    on_white = 1.05 / (luminance(rgb) + 0.05)
    on_ink = (luminance(rgb) + 0.05) / (luminance(ink) + 0.05)
    text = '#ffffff' if on_white >= on_ink else theme['ink']
    return tuple(c / 255 for c in rgb), text


def first_index(rows, test):
    return next(i for i, r in enumerate(rows) if test(r))


def events(t, rows):
    """Bird detection, the FSM's reaction and the bird's arrival had the drone not escaped.

    The bird picture moves east-west on the camera's north coordinate, so its horizontal
    distance to the camera is the east distance between the picture centre (rig start +
    rig speed x t_rel) and the drone (x_east). Without an escape the drone would have
    stayed where it was at the ESCAPE tick.
    """
    escape = first_index(rows, lambda r: r['state'] == 'ESCAPE')
    start = max(i for i, ti in enumerate(t) if ti <= 0.0)  # last tick before the rig start
    vx = number(rows[escape]['rig_cmd_vx'])
    x0 = number(rows[start]['x_east']) + bird_start()
    bird_t = [0.0] + [ti for ti in t if 0.0 < ti <= t[escape]]
    bird_d = [x0 + vx * ti - number(r['x_east'])
              for ti, r in zip(bird_t, [rows[start]] + rows[start + 1:escape + 1])]
    ev = {
        'detect': t[first_index(rows, lambda r: number(r['yolo_hazard']) == 1)],
        'warn': t[first_index(rows, lambda r: r['state'] == 'EARLY_WARNING')],
        'escape': t[escape],
        'd_escape': bird_d[-1],
        'v_bird': abs(vx),
        'bird_t': bird_t,
        'bird_d': bird_d,
    }
    ev['arrive'] = ev['escape'] + ev['d_escape'] / ev['v_bird']
    return ev


def event_lines(ev):
    """One vertical line per event time; events in the same tick share a line and label."""
    lines = []
    for x, name in sorted(((ev['detect'], 'bird detected'), (ev['warn'], 'EARLY_WARNING'),
                           (ev['escape'], 'ESCAPE command'),
                           (ev['arrive'],
                            'bird reaches the drone (picture centre over the drone)')),
                          key=lambda e: e[0]):
        if lines and math.isclose(lines[-1][0], x):
            lines[-1][1].append(name)
        else:
            lines.append([x, [name]])
    return lines


def style_axis(ax, theme, ylabel):
    ax.set_facecolor(theme['surface'])
    ax.grid(axis='y', color=theme['grid'], linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(theme['axis'])
    ax.tick_params(colors=theme['muted'], labelsize=8, length=0)
    ax.set_ylabel(ylabel, color=theme['ink2'], fontsize=9)


def threshold(ax, y, label, theme, x_label, below=False):
    ax.axhline(y, color=theme['muted'], linewidth=0.8, linestyle=(0, (4, 3)))
    ax.text(x_label, y, f' {label}', color=theme['ink2'], fontsize=7.5,
            va='top' if below else 'bottom', ha='left')


def line(ax, t, values, colour, label, theme, label_x=None, ha='left', dy=0.0):
    """Plot one series; label it directly at label_x where the series are apart."""
    ax.plot(t, values, color=colour, linewidth=1.6, label=label, solid_capstyle='round')
    if label_x is None:
        return
    i = min(range(len(t)), key=lambda k: abs(t[k] - label_x))
    if not math.isnan(values[i]):
        ax.text(label_x, values[i] + dy, label, color=theme['ink2'], fontsize=7.5,
                ha=ha, va='center')


def first_above(t, values, level):
    return next((ti for ti, v in zip(t, values) if v > level), None)


def bracket(ax, x0, x1, y, label, theme):
    """Interval bracket with its length above and its name below."""
    ax.annotate('', xy=(x1, y), xytext=(x0, y),
                arrowprops={'arrowstyle': '|-|,widthA=0.3,widthB=0.3', 'color': theme['ink2'],
                            'linewidth': 0.9, 'shrinkA': 0, 'shrinkB': 0})
    mid = (x0 + x1) / 2
    ax.text(mid, y + 0.08, f'{x1 - x0:.2f} s', color=theme['ink'], fontsize=8.5,
            fontweight='bold', ha='center', va='bottom')
    # Surface-coloured box: an event line next to a wide name passes behind it.
    ax.text(mid, y - 0.08, label, color=theme['ink2'], fontsize=7.5, ha='center', va='top',
            zorder=4, bbox={'facecolor': theme['surface'], 'edgecolor': 'none', 'pad': 0.5})


def event_lane(ax, theme, lines, ev):
    ax.set_facecolor(theme['surface'])
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    for k, (x, names) in enumerate(lines):
        # The first label sits left of its line, the later ones right of theirs.
        ha, dx = ('right', -0.08) if k == 0 else ('left', 0.08)
        ax.text(x + dx, 0.98, ' + '.join(names), color=theme['ink'], fontsize=7.5,
                ha=ha, va='top')
        ax.text(x + dx, 0.76, f'{x:.2f} s', color=theme['ink2'], fontsize=7.5,
                ha=ha, va='top')
    bracket(ax, ev['detect'], ev['escape'], 0.22, 'detection → escape', theme)
    bracket(ax, ev['escape'], ev['arrive'], 0.22, 'escape margin', theme)


def plot(path, output):
    t, rows = load(path)
    segs = segments(t, rows)
    transitions = [s[1] for s in segs[1:]]
    ev = events(t, rows)
    lines = event_lines(ev)
    theme = THEME
    fig, axes = plt.subplots(
        5, 1, figsize=(9.0, 8.9), sharex=True, dpi=150,
        gridspec_kw={'height_ratios': [1.1, 3, 3, 3, 1.1], 'hspace': 0.28})
    fig.patch.set_facecolor(theme['surface'])
    x_label = t[0]

    event_lane(axes[0], theme, lines, ev)

    ax = axes[1]
    style_axis(ax, theme, 'distance to the drone [m]')
    line(ax, t, [number(r['lidar_m']) for r in rows], theme['series'][0],
         'closest object (LiDAR, pole surface)', theme)
    line(ax, ev['bird_t'], ev['bird_d'], theme['series'][1],
         'bird picture centre', theme)
    # Without an escape the bird keeps its speed from the ESCAPE tick until d = 0.
    ax.plot([ev['escape'], ev['arrive']], [ev['d_escape'], 0.0], color=theme['series'][1],
            linewidth=1.6, linestyle=(0, (2, 2)),
            label='bird without escape (extrapolated)')
    ax.text(ev['arrive'] + 0.1, 0.12, f'0 m at {ev["arrive"]:.2f} s', color=theme['ink2'],
            fontsize=7.5, ha='left', va='bottom')
    threshold(ax, 3.0, 'D_warn 3.0 m', theme, x_label)
    threshold(ax, 1.5, 'D_confirm 1.5 m', theme, x_label)
    threshold(ax, 1.2, 'D_hazard 1.2 m', theme, x_label, below=True)
    ax.set_ylim(0, 8)
    # Right of the LiDAR line (it leaves the panel at the top) and above the D_warn line.
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 0.8), fontsize=7.5, frameon=False,
              labelcolor=theme['ink2'])

    ax = axes[2]
    style_axis(ax, theme, 'score')
    conf = [number(r['yolo_conf']) for r in rows]
    fusion = [number(r['fusion']) for r in rows]
    rise = first_above(t, fusion, 0.3)
    # Label both series left of the rise; push the labels apart if the values are close.
    dy_conf = dy_fusion = 0.0
    if rise is not None:
        i = t.index(rise)
        gap = conf[i] - fusion[i]
        if abs(gap) < 0.12:
            dy_conf = 0.07 if gap >= 0 else -0.07
            dy_fusion = -dy_conf
    label_x = rise - 0.1 if rise is not None else None
    line(ax, t, conf, theme['series'][0], 'YOLO conf', theme,
         label_x=label_x, ha='right', dy=dy_conf)
    line(ax, t, fusion, theme['series'][1], 'fusion', theme,
         label_x=label_x, ha='right', dy=dy_fusion)
    for y, label in ((0.35, 'EARLY_WARNING 0.35'), (0.65, 'THREAT_CONFIRMED 0.65')):
        threshold(ax, y, label, theme, x_label)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', fontsize=7.5, frameon=False, labelcolor=theme['ink2'],
              ncol=2)

    ax = axes[3]
    style_axis(ax, theme, 'drone velocity [m/s]')
    escape = [s for s in segs if s[0] == 'ESCAPE']
    plateau = escape[0][2] - 1.0 if escape else None
    for key, label, colour, dy in (('v_east', 'east', theme['series'][0], 0.35),
                                   ('v_north', 'north', theme['series'][1], 0.35),
                                   ('v_up', 'up', theme['series'][2], -0.35)):
        line(ax, t, [number(r[key]) for r in rows], colour, label, theme,
             label_x=plateau, dy=dy)
    ax.axhline(0, color=theme['axis'], linewidth=0.8)
    ax.legend(loc='upper left', fontsize=7.5, frameon=False, labelcolor=theme['ink2'],
              ncol=3)

    ax = axes[4]
    ax.set_facecolor(theme['surface'])
    ax.set_yticks([])
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(theme['axis'])
    ax.tick_params(colors=theme['muted'], labelsize=8, length=0)
    for state, x0, x1 in segs:
        fill, ink = state_fill(state, theme)
        # 2 px surface gap between neighbouring segments
        ax.broken_barh([(x0, max(x1 - x0 - 0.03, 0.01))], (0, 1), facecolors=fill)
        if x1 - x0 > 1.3:
            ax.text((x0 + x1) / 2, 0.5, SHORT.get(state, state), color=ink, fontsize=7,
                    ha='center', va='center')
        else:  # too short for an inside label: name it above the band
            ax.text((x0 + x1) / 2, 1.08, state, color=theme['ink2'], fontsize=7,
                    ha='center', va='bottom', zorder=4,
                    bbox={'facecolor': theme['surface'], 'edgecolor': 'none', 'pad': 0.5})
    ax.set_ylim(0, 1.6)
    ax.set_xlabel('time since the raptor rig started moving [s] (simulation time)',
                  color=theme['ink2'], fontsize=9)

    for ax in axes[1:4]:
        for x in transitions:
            ax.axvline(x, color=theme['grid'], linewidth=0.9, zorder=0)
    for ax in axes:
        for x, _ in lines:
            ax.axvline(x, color=theme['ink2'], linewidth=0.8, linestyle=EVENT_DASH,
                       zorder=3)
    axes[0].set_xlim(t[0], t[-1] + 0.3)
    fig.suptitle("Raptor scenario in simulation: the state machine's reaction and the "
                 "bird's arrival", color=theme['ink'], fontsize=11, x=0.08, ha='left',
                 y=0.97)
    fig.text(0.08, 0.937, 'Gazebo Harmonic + ArduPilot SITL + ROS 2 Jazzy; bird picture '
             f'on a pole at {ev["v_bird"]:g} m/s; grey lines: state changes, dashed: '
             'events', color=theme['muted'], fontsize=8, ha='left')
    fig.subplots_adjust(left=0.08, right=0.97, top=0.915, bottom=0.07)
    fig.savefig(output, facecolor=theme['surface'])
    plt.close(fig)
    print('wrote', output)


if __name__ == '__main__':
    plot(sys.argv[1], sys.argv[2])
