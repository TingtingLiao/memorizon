"""Discrete keyboard actions -> camera trajectory.

One action per latent frame (4 per second of video). An action is any combination of

    w / s   move forward / backward        a / d   move left / right
    j / l   turn left / right              i / k   look up / down
    .       stay still

Actions are written as space-separated tokens, each optionally repeated with ``*N``:

    "w*16 wl*8 w*16 .*4 l*24"   # 4 s forward, 2 s forward while turning right, ...

The step sizes are those of the training data: a move is 0.25 m (1 m/s) along the
horizontal heading, a turn or look is 7.5 degrees. When moving while turning, the step
follows the heading halfway through the turn. Sideways moves (a/d) are rare in the
training data, so they are followed less reliably than w/s.
"""

from __future__ import annotations

import numpy as np

MOVE_STEP = 0.25 / 4           # metres per latent frame, in trajectory units (metres / 4)
TURN_STEP = np.radians(7.5)    # radians per latent frame

_MOVES = {"w": (0, 1), "s": (0, -1), "a": (-1, 0), "d": (1, 0)}   # (right, forward)
_TURNS = {"j": (-1, 0), "l": (1, 0), "i": (0, 1), "k": (0, -1)}   # (yaw right, pitch up)


def parse_actions(text: str) -> list[str]:
    """``"w*3 wl ."`` -> ``["w", "w", "w", "wl", "."]``."""
    steps = []
    for token in text.split():
        keys, _, n = token.partition("*")
        bad = set(keys) - set(_MOVES) - set(_TURNS) - {"."}
        if bad:
            raise ValueError(f"unknown action key(s) {sorted(bad)} in {token!r}")
        steps += [keys] * (int(n) if n else 1)
    return steps


def _rotation(yaw: float, pitch: float) -> np.ndarray:
    """Camera-to-world rotation, OpenCV axes (x right, y down, z forward)."""
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    r_yaw = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    r_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return r_yaw @ r_pitch


def actions_to_c2w(actions: str | list[str]) -> np.ndarray:
    """``[T, 4, 4]`` camera-to-world poses, ``T = len(actions) + 1``, starting at identity."""
    steps = parse_actions(actions) if isinstance(actions, str) else actions
    yaw = pitch = 0.0
    pos = np.zeros(3)
    poses = [np.eye(4)]
    for keys in steps:
        d_yaw = sum(_TURNS[k][0] for k in keys if k in _TURNS) * TURN_STEP
        d_pitch = sum(_TURNS[k][1] for k in keys if k in _TURNS) * TURN_STEP
        right = sum(_MOVES[k][0] for k in keys if k in _MOVES)
        fwd = sum(_MOVES[k][1] for k in keys if k in _MOVES)
        if right or fwd:
            h = yaw + d_yaw / 2                                    # heading halfway through the turn
            move = fwd * np.array([np.sin(h), 0, np.cos(h)]) + right * np.array([np.cos(h), 0, -np.sin(h)])
            pos = pos + MOVE_STEP * move / np.linalg.norm(move)
        yaw += d_yaw
        pitch = float(np.clip(pitch + d_pitch, -np.pi / 3, np.pi / 3))
        pose = np.eye(4)
        pose[:3, :3] = _rotation(yaw, pitch)
        pose[:3, 3] = pos
        poses.append(pose)
    return np.stack(poses).astype(np.float32)
