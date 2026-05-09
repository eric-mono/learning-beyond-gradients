# Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Clone-state graph search for hand-written Montezuma primitive policies.

This is still a pure heuristic search: every node stores one ALE system-state
snapshot and a replayable primitive-action suffix after a deterministic warmup.
Each depth expansion restores one parent snapshot, executes one Atari action,
and keeps newly discovered RAM states under a spatially bucketed frontier cap.

Compared with the EnvPool replay-based graph search, this avoids replaying the
entire prefix from reset for every candidate branch, which makes deeper
single-step exploration much cheaper. Every depth appends one JSONL record and
rewrites a summary CSV with cumulative step counts for sample-efficiency plots.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any

from ale_py import ALEInterface, roms


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_PATH = SCRIPT_DIR / "heuristic_montezuma_ale_state_trials.jsonl"
DEFAULT_SUMMARY_PATH = (
    SCRIPT_DIR / "heuristic_montezuma_ale_state_trials_summary.csv"
)


class MontezumaAction(IntEnum):
    NOOP = 0
    FIRE = 1
    UP = 2
    RIGHT = 3
    LEFT = 4
    DOWN = 5
    UPRIGHT = 6
    UPLEFT = 7
    DOWNRIGHT = 8
    DOWNLEFT = 9
    UPFIRE = 10
    RIGHTFIRE = 11
    LEFTFIRE = 12
    DOWNFIRE = 13
    UPRIGHTFIRE = 14
    UPLEFTFIRE = 15
    DOWNRIGHTFIRE = 16
    DOWNLEFTFIRE = 17


@dataclass(frozen=True)
class MontezumaState:
    player_x: int
    player_y: int
    motion_state: int
    platform_state: int
    skull_x: int
    room_id: int
    lives: int


@dataclass(frozen=True)
class FrontierNode:
    sequence: tuple[int, ...]
    ale_state: object
    state: MontezumaState
    score: float
    key_distance: int
    state_key: bytes


def append_trial_record(log_path: Path, record: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def load_trial_records(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    records = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_summary(
    log_path: Path,
    summary_path: Path,
) -> list[dict[str, Any]]:
    records = load_trial_records(log_path)
    rows = []
    cumulative_env_steps = 0
    cumulative_ale_frames = 0
    best_score_so_far = float("-inf")
    best_key_distance_so_far = None
    for index, record in enumerate(records):
        env_steps = int(record.get("env_steps", 0))
        ale_frames = int(record.get("ale_frames", 0))
        cumulative_env_steps += env_steps
        cumulative_ale_frames += ale_frames
        score_max = float(record.get("score_max", 0.0))
        best_score_so_far = max(best_score_so_far, score_max)
        key_distance = record.get("best_key_distance")
        if key_distance is not None:
            key_distance = int(key_distance)
            if best_key_distance_so_far is None:
                best_key_distance_so_far = key_distance
            else:
                best_key_distance_so_far = min(
                    best_key_distance_so_far,
                    key_distance,
                )
        rows.append(
            {
                "trial_index": index,
                "timestamp": record.get("timestamp", ""),
                "trial_name": record.get("trial_name", ""),
                "depth": record.get("depth", ""),
                "num_candidates": record.get("num_candidates", ""),
                "frontier_limit": record.get("frontier_limit", ""),
                "episodes_started": record.get("episodes_started", ""),
                "episodes_finished": record.get("episodes_finished", ""),
                "env_steps": env_steps,
                "ale_frames": ale_frames,
                "cumulative_env_steps": cumulative_env_steps,
                "cumulative_ale_frames": cumulative_ale_frames,
                "score_max": score_max,
                "best_score_so_far": best_score_so_far,
                "best_room_id": record.get("best_room_id", ""),
                "best_key_distance": record.get("best_key_distance", ""),
                "best_key_distance_so_far": best_key_distance_so_far,
                "new_states": record.get("new_states", ""),
                "frontier_size": record.get("frontier_size", ""),
                "visited_states": record.get("visited_states", ""),
                "best_sequence": record.get("best_sequence", ""),
            }
        )

    fieldnames = [
        "trial_index",
        "timestamp",
        "trial_name",
        "depth",
        "num_candidates",
        "frontier_limit",
        "episodes_started",
        "episodes_finished",
        "env_steps",
        "ale_frames",
        "cumulative_env_steps",
        "cumulative_ale_frames",
        "score_max",
        "best_score_so_far",
        "best_room_id",
        "best_key_distance",
        "best_key_distance_so_far",
        "new_states",
        "frontier_size",
        "visited_states",
        "best_sequence",
    ]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def print_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("trial_log_summary: records=0 cumulative_env_steps=0")
        return
    last_row = rows[-1]
    best_row = max(
        rows,
        key=lambda row: (
            float(row["best_score_so_far"]),
            -999999
            if row["best_key_distance_so_far"] in ("", None)
            else -int(row["best_key_distance_so_far"]),
        ),
    )
    print(
        "trial_log_summary:",
        f"records={len(rows)}",
        f"cumulative_env_steps={last_row['cumulative_env_steps']}",
        f"cumulative_ale_frames={last_row['cumulative_ale_frames']}",
        f"visited_states={last_row['visited_states']}",
    )
    print(
        "best_progress:",
        f"trial_index={best_row['trial_index']}",
        f"trial_name={best_row['trial_name']}",
        f"best_score_so_far={best_row['best_score_so_far']:.1f}",
        f"best_key_distance_so_far={best_row['best_key_distance_so_far']}",
        f"best_sequence={best_row['best_sequence']}",
    )


def parse_action_set(spec: str) -> tuple[int, ...]:
    actions = []
    for item in spec.split(","):
        item = item.strip()
        if item:
            actions.append(int(MontezumaAction[item]))
    if not actions:
        raise ValueError("action set must not be empty")
    return tuple(actions)


def parse_warmup_actions(spec: str) -> tuple[int, ...]:
    actions = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            name, count = item.split(":", 1)
            actions.extend([int(MontezumaAction[name.strip()])] * int(count))
        else:
            actions.append(int(MontezumaAction[item]))
    return tuple(actions)


def sequence_to_repr(
    warmup_actions: tuple[int, ...],
    sequence: tuple[int, ...],
) -> list[str]:
    return [MontezumaAction(action).name for action in (*warmup_actions, *sequence)]


def decode_ale_state(ale: ALEInterface) -> MontezumaState:
    ram = ale.getRAM()
    return MontezumaState(
        player_x=int(ram[42]),
        player_y=(312 - int(ram[43])) & 255,
        motion_state=int(ram[2]),
        platform_state=int(ram[89]),
        skull_x=int(ram[47]),
        room_id=int(ram[3]),
        lives=int(ale.lives()),
    )


def canonical_state_key(
    ale: ALEInterface,
    *,
    ignore_ram_bytes: tuple[int, ...],
) -> bytes:
    ram = bytearray(int(value) for value in ale.getRAM())
    for index in ignore_ram_bytes:
        ram[index] = 0
    ram.append(int(ale.lives()))
    return bytes(ram)


def target_distance(
    state: MontezumaState,
    *,
    target_room: int,
    target_x: int,
    target_y: int,
) -> int:
    if state.room_id != target_room:
        return -100000 + abs(state.player_x - target_x) + abs(state.player_y - target_y)
    return abs(state.player_x - target_x) + abs(state.player_y - target_y)


def node_rank(node: FrontierNode) -> tuple[float, int, int, int, int, int]:
    state = node.state
    stable_bonus = 1 if state.motion_state == 4 and state.platform_state != 0 else 0
    no_fall_bonus = 1 if state.motion_state != 6 else 0
    return (
        node.score,
        1 if state.room_id != 1 else 0,
        state.lives,
        -node.key_distance,
        stable_bonus,
        no_fall_bonus,
    )


def select_frontier(
    nodes: list[FrontierNode],
    *,
    frontier_limit: int,
    x_bucket_size: int,
    y_bucket_size: int,
    per_bucket_limit: int,
) -> list[FrontierNode]:
    if len(nodes) <= frontier_limit:
        return sorted(nodes, key=node_rank, reverse=True)

    sorted_nodes = sorted(nodes, key=node_rank, reverse=True)
    selected = []
    bucket_counts = {}
    selected_keys = set()

    for node in sorted_nodes:
        bucket = (
            node.state.room_id,
            node.state.lives,
            node.state.motion_state,
            node.state.player_x // x_bucket_size,
            node.state.player_y // y_bucket_size,
        )
        if bucket_counts.get(bucket, 0) >= per_bucket_limit:
            continue
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        selected.append(node)
        selected_keys.add(node.state_key)
        if len(selected) >= frontier_limit:
            return selected

    for node in sorted_nodes:
        if node.state_key in selected_keys:
            continue
        selected.append(node)
        if len(selected) >= frontier_limit:
            break
    return selected


def build_ale(seed: int, repeat_action_probability: float) -> ALEInterface:
    ale = ALEInterface()
    ale.setInt("random_seed", seed)
    ale.setFloat("repeat_action_probability", repeat_action_probability)
    ale.setInt("frame_skip", 1)
    ale.loadROM(roms.get_rom_path("montezuma_revenge"))
    return ale


def initialize_root(
    ale: ALEInterface,
    *,
    warmup_actions: tuple[int, ...],
    ignore_ram_bytes: tuple[int, ...],
    target_room: int,
    target_x: int,
    target_y: int,
) -> tuple[FrontierNode, int]:
    ale.reset_game()
    env_steps = 0
    for action in warmup_actions:
        ale.act(action)
        env_steps += 1
    state = decode_ale_state(ale)
    state_key = canonical_state_key(ale, ignore_ram_bytes=ignore_ram_bytes)
    node = FrontierNode(
        sequence=(),
        ale_state=ale.cloneSystemState(),
        state=state,
        score=0.0,
        key_distance=target_distance(
            state,
            target_room=target_room,
            target_x=target_x,
            target_y=target_y,
        ),
        state_key=state_key,
    )
    return node, env_steps


def expand_frontier(
    ale: ALEInterface,
    frontier: list[FrontierNode],
    *,
    visited_scores: dict[bytes, float],
    action_set: tuple[int, ...],
    ignore_ram_bytes: tuple[int, ...],
    target_room: int,
    target_x: int,
    target_y: int,
    min_lives: int,
) -> tuple[list[FrontierNode], dict[str, Any]]:
    next_nodes = []
    num_candidates = len(frontier) * len(action_set)
    episodes_finished = 0
    score_max = float("-inf")
    best_state = None
    best_sequence = ()
    best_room_id = 1
    best_key_distance = None
    best_key_state = None
    best_key_sequence = ()

    for parent in frontier:
        for action in action_set:
            ale.restoreSystemState(parent.ale_state)
            reward = float(ale.act(action))
            sequence = parent.sequence + (action,)
            score = parent.score + reward
            state = decode_ale_state(ale)
            is_terminal = ale.game_over()
            if is_terminal or state.lives < parent.state.lives:
                episodes_finished += 1

            if score > score_max:
                score_max = score
                best_state = state
                best_sequence = sequence
                best_room_id = state.room_id

            key_distance = target_distance(
                state,
                target_room=target_room,
                target_x=target_x,
                target_y=target_y,
            )
            if state.lives >= min_lives and (
                best_key_distance is None or key_distance < best_key_distance
            ):
                best_key_distance = key_distance
                best_key_state = state
                best_key_sequence = sequence

            if is_terminal or state.lives < min_lives:
                continue

            state_key = canonical_state_key(
                ale,
                ignore_ram_bytes=ignore_ram_bytes,
            )
            prev_score = visited_scores.get(state_key)
            if prev_score is not None and prev_score >= score:
                continue
            visited_scores[state_key] = score
            next_nodes.append(
                FrontierNode(
                    sequence=sequence,
                    ale_state=ale.cloneSystemState(),
                    state=state,
                    score=score,
                    key_distance=key_distance,
                    state_key=state_key,
                )
            )

    metrics = {
        "num_candidates": num_candidates,
        "episodes_finished": episodes_finished,
        "score_max": score_max,
        "best_state": best_state,
        "best_sequence": best_sequence,
        "best_room_id": best_room_id,
        "best_key_distance": best_key_distance,
        "best_key_state": best_key_state,
        "best_key_sequence": best_key_sequence,
        "env_steps": num_candidates,
        "ale_frames": num_candidates,
        "new_states": len(next_nodes),
    }
    return next_nodes, metrics


def run_search(args: argparse.Namespace) -> None:
    action_set = parse_action_set(args.action_set)
    warmup_actions = parse_warmup_actions(args.warmup_actions)
    ignore_ram_bytes = tuple(args.ignore_ram_bytes)

    ale = build_ale(
        seed=args.seed,
        repeat_action_probability=args.repeat_action_probability,
    )
    root, warmup_steps = initialize_root(
        ale,
        warmup_actions=warmup_actions,
        ignore_ram_bytes=ignore_ram_bytes,
        target_room=args.target_room,
        target_x=args.target_x,
        target_y=args.target_y,
    )

    visited_scores = {root.state_key: root.score}
    frontier = [root]
    if args.frontier_limit < 1:
        raise ValueError("frontier_limit must be positive")
    if args.x_bucket_size < 1 or args.y_bucket_size < 1:
        raise ValueError("bucket sizes must be positive")
    if args.per_bucket_limit < 1:
        raise ValueError("per_bucket_limit must be positive")

    for depth in range(1, args.depth + 1):
        next_nodes, metrics = expand_frontier(
            ale,
            frontier,
            visited_scores=visited_scores,
            action_set=action_set,
            ignore_ram_bytes=ignore_ram_bytes,
            target_room=args.target_room,
            target_x=args.target_x,
            target_y=args.target_y,
            min_lives=args.min_lives,
        )
        frontier = select_frontier(
            next_nodes,
            frontier_limit=args.frontier_limit,
            x_bucket_size=args.x_bucket_size,
            y_bucket_size=args.y_bucket_size,
            per_bucket_limit=args.per_bucket_limit,
        )

        env_steps = int(metrics["env_steps"])
        ale_frames = int(metrics["ale_frames"])
        if depth == 1:
            env_steps += warmup_steps
            ale_frames += warmup_steps

        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "trial_name": args.trial_name,
            "kind": "search",
            "depth": depth,
            "num_candidates": int(metrics["num_candidates"]),
            "frontier_limit": args.frontier_limit,
            "frontier_size": len(frontier),
            "new_states": int(metrics["new_states"]),
            "visited_states": len(visited_scores),
            "episodes_started": int(metrics["num_candidates"]) + (1 if depth == 1 else 0),
            "episodes_finished": int(metrics["episodes_finished"]),
            "env_steps": env_steps,
            "ale_frames": ale_frames,
            "score_max": float(metrics["score_max"]),
            "best_room_id": int(metrics["best_room_id"]),
            "best_state": None
            if metrics["best_state"] is None
            else metrics["best_state"].__dict__,
            "best_sequence": sequence_to_repr(
                warmup_actions,
                metrics["best_sequence"],
            ),
            "best_key_distance": metrics["best_key_distance"],
            "best_key_state": None
            if metrics["best_key_state"] is None
            else metrics["best_key_state"].__dict__,
            "best_key_sequence": sequence_to_repr(
                warmup_actions,
                metrics["best_key_sequence"],
            ),
            "warmup_actions": sequence_to_repr(warmup_actions, ()),
            "action_set": [MontezumaAction(action).name for action in action_set],
            "ignore_ram_bytes": list(ignore_ram_bytes),
            "seed": args.seed,
            "repeat_action_probability": args.repeat_action_probability,
            "target_room": args.target_room,
            "target_x": args.target_x,
            "target_y": args.target_y,
            "min_lives": args.min_lives,
            "x_bucket_size": args.x_bucket_size,
            "y_bucket_size": args.y_bucket_size,
            "per_bucket_limit": args.per_bucket_limit,
            "notes": args.notes,
        }
        append_trial_record(args.log_path, record)
        print(
            "depth_result:",
            f"depth={depth}",
            f"num_candidates={record['num_candidates']}",
            f"new_states={record['new_states']}",
            f"frontier_size={record['frontier_size']}",
            f"visited_states={record['visited_states']}",
            f"score_max={record['score_max']:.1f}",
            f"best_room_id={record['best_room_id']}",
            f"best_key_distance={record['best_key_distance']}",
            f"best_state={record['best_state']}",
            f"best_key_state={record['best_key_state']}",
            f"best_key_sequence={record['best_key_sequence']}",
        )
        if not frontier:
            break

    rows = write_summary(args.log_path, args.summary_path)
    print_summary(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone-state Montezuma heuristic graph search.",
    )
    parser.add_argument("--depth", type=int, default=96)
    parser.add_argument("--frontier-limit", type=int, default=4096)
    parser.add_argument("--x-bucket-size", type=int, default=8)
    parser.add_argument("--y-bucket-size", type=int, default=8)
    parser.add_argument("--per-bucket-limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat-action-probability", type=float, default=0.0)
    parser.add_argument("--warmup-actions", type=str, default="FIRE,NOOP:20")
    parser.add_argument(
        "--action-set",
        type=str,
        default=(
            "NOOP,FIRE,UP,RIGHT,LEFT,DOWN,UPRIGHT,UPLEFT,"
            "DOWNRIGHT,DOWNLEFT,UPFIRE,RIGHTFIRE,LEFTFIRE,"
            "DOWNFIRE,UPRIGHTFIRE,UPLEFTFIRE,DOWNRIGHTFIRE,DOWNLEFTFIRE"
        ),
    )
    parser.add_argument("--ignore-ram-bytes", type=int, nargs="*", default=[102])
    parser.add_argument("--target-room", type=int, default=1)
    parser.add_argument("--target-x", type=int, default=8)
    parser.add_argument("--target-y", type=int, default=104)
    parser.add_argument("--min-lives", type=int, default=6)
    parser.add_argument(
        "--trial-name",
        type=str,
        default="montezuma_ale_state_search",
    )
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG_PATH,
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_search(parse_args())
