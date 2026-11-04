#!/usr/bin/env python3
# Copyright 2026 Aycan Coskun
# SPDX-License-Identifier: AGPL-3.0-only

"""
Plot a raptor scenario run exported with `ros2 run raptor_fsm export_run_csv`.

Four panels on a shared time axis (t = 0 when the raptor rig starts moving): LiDAR
distance with the paper's distance thresholds, YOLO confidence and fusion score with the
FSM thresholds, the drone's velocity, and the state timeline. Writes a light and a dark
PNG. Usage: plot_scenario.py <run.csv> <output_prefix>
"""

import csv
import math
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

THEMES = {
    'light': {
        'surface': '#fcfcfb', 'ink': '#0b0b0b', 'ink2': '#52514e', 'muted': '#898781',
        'grid': '#e1e0d9', 'axis': '#c3c2b7',
        'series': ['#2a78d6', '#eb6834', '#1baf7a', '#eda100'],
        'on_fill': ['#ffffff', '#ffffff', '#ffffff', '#0b0b0b'], 'idle': '#c3c2b7',
    },
    'dark': {
        'surface': '#1a1a19', 'ink': '#f0efec', 'ink2': '#c3c2b7', 'muted': '#898781',
        'grid': '#2c2c2a', 'axis': '#383835',
        'series': ['#3987e5', '#d95926', '#199e70', '#c98500'],
        'on_fill': ['#ffffff', '#ffffff', '#ffffff', '#ffffff'], 'idle': '#383835',
    },
}
RAPTOR_STATES = ['EARLY_WARNING', 'THREAT_CONFIRMED', 'ESCAPE', 'RTL']  # fixed slot order
SHORT = {'IDLE': 'IDLE', 'EARLY_WARNING': 'EARLY_WARNING', 'THREAT_CONFIRMED': 'THREAT_CONF.',
         'ESCAPE': 'ESCAPE', 'RTL': 'RTL', 'OBSTACLE_AVOID': 'OBSTACLE_AVOID'}


def number(text):
    try:
        value = float(text)
    except ValueError:
        return math.nan
    return value if math.isfinite(value) else math.nan


def load(path):
    rows = list(csv.DictReader(open(path)))
    start = next(float(r['t_sim']) for r in rows if number(r['rig_cmd_vx']) not in (0.0,)
                 and not math.isnan(number(r['rig_cmd_vx'])))
    rtl = next(float(r['t_sim']) for r in rows if r['state'] == 'RTL')
    window = [r for r in rows if start - 1.0 <= float(r['t_sim']) <= rtl + 3.0]
    t = [float(r['t_sim']) - start for r in window]
    return t, window


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


def plot(path, prefix):
    t, rows = load(path)
    segs = segments(t, rows)
    transitions = [s[1] for s in segs[1:]]
    for name, theme in THEMES.items():
        fig, axes = plt.subplots(
            4, 1, figsize=(9.0, 7.6), sharex=True, dpi=150,
            gridspec_kw={'height_ratios': [3, 3, 3, 1.1], 'hspace': 0.28})
        fig.patch.set_facecolor(theme['surface'])
        x_label = t[0]

        ax = axes[0]
        style_axis(ax, theme, 'LiDAR distance [m]')
        line(ax, t, [number(r['lidar_m']) for r in rows], theme['series'][0], 'closest object',
             theme)
        threshold(ax, 3.0, 'D_warn 3.0 m', theme, x_label)
        threshold(ax, 1.5, 'D_confirm 1.5 m', theme, x_label)
        threshold(ax, 1.2, 'D_hazard 1.2 m', theme, x_label, below=True)
        ax.set_ylim(0, 8)

        ax = axes[1]
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

        ax = axes[2]
        style_axis(ax, theme, 'drone velocity [m/s]')
        escape = [s for s in segs if s[0] == 'ESCAPE']
        plateau = escape[0][2] - 1.0 if escape else None
        for key, label, colour, dy in (('v_east', 'east', theme['series'][0], -0.35),
                                       ('v_north', 'north', theme['series'][1], 0.35),
                                       ('v_up', 'up', theme['series'][2], -0.35)):
            line(ax, t, [number(r[key]) for r in rows], colour, label, theme,
                 label_x=plateau, dy=dy)
        ax.axhline(0, color=theme['axis'], linewidth=0.8)
        ax.legend(loc='upper left', fontsize=7.5, frameon=False, labelcolor=theme['ink2'],
                  ncol=3)

        ax = axes[3]
        ax.set_facecolor(theme['surface'])
        ax.set_yticks([])
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(theme['axis'])
        ax.tick_params(colors=theme['muted'], labelsize=8, length=0)
        for state, x0, x1 in segs:
            if state in RAPTOR_STATES:
                slot = RAPTOR_STATES.index(state)
                fill, ink = theme['series'][slot], theme['on_fill'][slot]
            else:
                fill, ink = theme['idle'], theme['ink']
            # 2 px surface gap between neighbouring segments
            ax.broken_barh([(x0, max(x1 - x0 - 0.03, 0.01))], (0, 1), facecolors=fill)
            if x1 - x0 > 1.3:
                ax.text((x0 + x1) / 2, 0.5, SHORT.get(state, state), color=ink, fontsize=7,
                        ha='center', va='center')
            else:  # too short for an inside label: name it above the band
                ax.text((x0 + x1) / 2, 1.08, state, color=theme['ink2'], fontsize=7,
                        ha='center', va='bottom')
        ax.set_ylim(0, 1.6)
        ax.set_xlabel('time since the raptor rig started moving [s] (simulation time)',
                      color=theme['ink2'], fontsize=9)

        for ax in axes[:3]:
            for x in transitions:
                ax.axvline(x, color=theme['grid'], linewidth=0.9, zorder=0)
        axes[0].set_xlim(t[0], t[-1] + 0.3)
        fig.suptitle('Raptor scenario in simulation: state machine decisions and vehicle '
                     'response', color=theme['ink'], fontsize=11, x=0.08, ha='left', y=0.965)
        fig.text(0.08, 0.925, 'Gazebo Harmonic + ArduPilot SITL + ROS 2 Jazzy; bird picture '
                 'on a pole approaching at 0.5 m/s; vertical lines = state changes',
                 color=theme['muted'], fontsize=8, ha='left')
        fig.subplots_adjust(left=0.08, right=0.9, top=0.9, bottom=0.08)
        out = f'{prefix}_{name}.png'
        fig.savefig(out, facecolor=theme['surface'])
        plt.close(fig)
        print('wrote', out)


if __name__ == '__main__':
    plot(sys.argv[1], sys.argv[2])
