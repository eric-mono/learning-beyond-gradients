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

"""Stable-state graph search for deterministic Montezuma macro heuristics.

Each node in the graph is a replayable macro sequence from reset plus the
settled RAM state reached after a short NOOP tail. At every depth we expand the
frontier with one extra macro, keep only newly discovered stable states, and log
the number of sampled episodes/env-steps so the sample-efficiency curve is
directly recoverable from the JSONL/CSV files.
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

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_PATH = SCRIPT_DIR / "heuristic_montezuma_state_graph_trials.jsonl"
DEFAULT_SUMMARY_PATH = (
    SCRIPT_DIR / "heuristic_montezuma_state_graph_trials_summary.csv"
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
class Macro:
    action: int
    steps: int


@dataclass(frozen=True)
class Candidate:
    sequence: tuple[Macro, ...]


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
class CandidateResult:
    candidate: Candidate
    score: float
    episode_length: int
    final_state: MontezumaState
    stable_state: MontezumaState
    is_stable: bool
    best_target_distance: int
    ranking_score: float


def reset_env_with_info(env):
    result = env.reset()
    if isinstance(result, tuple):
        return result
    return result, {}


def step_env(env, actions: np.ndarray):
    result = env.step(actions)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = np.logical_or(terminated, truncated)
    else:
        obs, reward, done, info = result
    return obs, np.asarray(reward), np.asarray(done), info


def decode_state(info: dict[str, np.ndarray], env_index: int) -> MontezumaState:
    ram = np.asarray(info["ram"])[env_index]
    return MontezumaState(
        player_x=int(ram[42]),
        player_y=(312 - int(ram[43])) & 255,
        motion_state=int(ram[2]),
        platform_state=int(ram[89]),
        skull_x=int(ram[47]),
        room_id=int(ram[3]),
        lives=int(np.asarray(info["lives"])[env_index]),
    )


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
    for index, record in enumerate(records):
        env_steps = int(record.get("env_steps", 0))
        ale_frames = int(record.get("ale_frames", 0))
        cumulative_env_steps += env_steps
        cumulative_ale_frames += ale_frames
        score_max = float(record.get("score_max", 0.0))
        best_score_so_far = max(best_score_so_far, score_max)
        rows.append(
            {
                "trial_index": index,
                "timestamp": record.get("timestamp", ""),
                "trial_name": record.get("trial_name", ""),
                "depth": record.get("depth", ""),
                "num_candidates": record.get("num_candidates", ""),
                "episodes_started": record.get("episodes_started", ""),
                "episodes_finished": record.get("episodes_finished", ""),
                "env_steps": env_steps,
                "ale_frames": ale_frames,
                "cumulative_env_steps": cumulative_env_steps,
                "cumulative_ale_frames": cumulative_ale_frames,
                "score_max": score_max,
                "best_score_so_far": best_score_so_far,
                "new_states": record.get("new_states", ""),
                "frontier_size": record.get("frontier_size", ""),
                "visited_states": record.get("visited_states", ""),
                "best_target_distance": record.get("best_target_distance", ""),
                "best_sequence": record.get("best_sequence", ""),
            }
        )

    fieldnames = [
        "trial_index",
        "timestamp",
        "trial_name",
        "depth",
        "num_candidates",
        "episodes_started",
        "episodes_finished",
        "env_steps",
        "ale_frames",
        "cumulative_env_steps",
        "cumulative_ale_frames",
        "score_max",
        "best_score_so_far",
        "new_states",
        "frontier_size",
        "visited_states",
        "best_target_distance",
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
            -int(row["best_target_distance"])
            if row["best_target_distance"] not in ("", None)
            else -999999,
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
        f"best_target_distance={best_row['best_target_distance']}",
        f"best_sequence={best_row['best_sequence']}",
    )


def parse_macro_library(spec: str) -> tuple[Macro, ...]:
    macros = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"macro item must be ACTION:steps, got {item!r}")
        name, steps = item.split(":", 1)
        macros.append(Macro(int(MontezumaAction[name.strip()]), int(steps)))
    return tuple(macros)


def sequence_to_repr(sequence: tuple[Macro, ...]) -> list[list[Any]]:
    return [[MontezumaAction(macro.action).name, macro.steps] for macro in sequence]


def state_to_dict(state: MontezumaState) -> dict[str, int]:
    return {
        "player_x": state.player_x,
        "player_y": state.player_y,
        "motion_state": state.motion_state,
        "platform_state": state.platform_state,
        "skull_x": state.skull_x,
        "room_id": state.room_id,
        "lives": state.lives,
    }


def state_key(state: MontezumaState) -> tuple[int, ...]:
    return (
        state.room_id,
        state.lives,
        state.motion_state,
        state.platform_state,
        state.player_x,
        state.player_y,
        state.skull_x,
    )


def is_stable_node(state: MontezumaState, initial_lives: int) -> bool:
    return (
        state.lives == initial_lives
        and state.motion_state == 4
        and state.platform_state != 0
    )


def rank_result(
    score: float,
    is_stable: bool,
    final_state: MontezumaState,
    target_x: int,
    target_y: int,
    initial_lives: int,
) -> float:
    rank = 1_000_000.0 * score
    rank += 100_000.0 * float(final_state.room_id != 1)
    rank += 10_000.0 * float(is_stable)
    if final_state.lives == initial_lives:
        rank += 1000.0
    else:
        rank -= 1000.0 * float(initial_lives - final_state.lives)
    rank -= 4.0 * abs(float(final_state.player_x - target_x))
    rank -= 2.0 * abs(float(final_state.player_y - target_y))
    return rank


def evaluate_candidates(
    candidates: list[Candidate],
    batch_size: int,
    frame_skip: int,
    noop_max: int,
    seed: int,
    settle_steps: int,
    stable_window: int,
    target_x: int,
    target_y: int,
) -> tuple[list[CandidateResult], int, int]:
    if not candidates:
        return [], 0, 0

    import envpool

    results = []
    total_env_steps = 0
    for start in range(0, len(candidates), batch_size):
        chunk = candidates[start : start + batch_size]
        num_envs = len(chunk)
        timelines = [
            [
                macro.action
                for macro in candidate.sequence
                for _ in range(macro.steps)
            ]
            for candidate in chunk
        ]
        horizons = np.asarray(
            [len(timeline) + settle_steps for timeline in timelines],
            dtype=np.int64,
        )
        max_steps = max(int(horizons.max(initial=0)), 1)
        env = envpool.make_gym(
            "MontezumaRevenge-v5",
            num_envs=num_envs,
            batch_size=num_envs,
            num_threads=min(num_envs, 8),
            seed=seed,
            max_episode_steps=max_steps,
            stack_num=1,
            gray_scale=False,
            frame_skip=frame_skip,
            noop_max=noop_max,
            use_fire_reset=True,
            episodic_life=False,
            reward_clip=False,
            repeat_action_probability=0.0,
            full_action_space=True,
        )

        _, info = reset_env_with_info(env)
        initial_lives = np.asarray(info["lives"], dtype=np.int64).copy()
        episode_scores = np.zeros(num_envs, dtype=np.float64)
        episode_lengths = np.zeros(num_envs, dtype=np.int64)
        done = np.zeros(num_envs, dtype=bool)
        prev_states = [decode_state(info, i) for i in range(num_envs)]
        final_states = list(prev_states)
        stable_states = list(prev_states)
        is_stable = np.zeros(num_envs, dtype=bool)
        stable_noop_run = np.zeros(num_envs, dtype=np.int64)
        best_target_distance = np.full(num_envs, 999999, dtype=np.int64)

        for t in range(max_steps):
            active = np.logical_and(~done, t < horizons)
            if not np.any(active):
                break

            actions = np.zeros(num_envs, dtype=np.int32)
            for env_index, timeline in enumerate(timelines):
                if active[env_index] and t < len(timeline):
                    actions[env_index] = timeline[t]

            _, reward, step_done, info = step_env(env, actions)
            episode_scores[active] += reward[active]
            episode_lengths[active] += 1

            for env_index in range(num_envs):
                if not active[env_index]:
                    continue
                state = decode_state(info, env_index)
                final_states[env_index] = state
                if state.lives == int(initial_lives[env_index]):
                    distance = abs(state.player_x - target_x) + abs(
                        state.player_y - target_y
                    )
                    best_target_distance[env_index] = min(
                        best_target_distance[env_index],
                        distance,
                    )

                prev_state = prev_states[env_index]
                if (
                    actions[env_index] == int(MontezumaAction.NOOP)
                    and state.lives == prev_state.lives
                    and state.player_x == prev_state.player_x
                    and state.player_y == prev_state.player_y
                    and state.motion_state == 4
                    and state.platform_state != 0
                ):
                    stable_noop_run[env_index] += 1
                else:
                    stable_noop_run[env_index] = 0
                prev_states[env_index] = state

                if (
                    stable_noop_run[env_index] >= stable_window
                    and is_stable_node(state, int(initial_lives[env_index]))
                ):
                    stable_states[env_index] = state
                    is_stable[env_index] = True
                    done[env_index] = True
                elif bool(step_done[env_index]) or t + 1 >= horizons[env_index]:
                    done[env_index] = True

            if np.all(done):
                break

        total_env_steps += int(episode_lengths.sum())
        env.close()

        for env_index, candidate in enumerate(chunk):
            result = CandidateResult(
                candidate=candidate,
                score=float(episode_scores[env_index]),
                episode_length=int(episode_lengths[env_index]),
                final_state=final_states[env_index],
                stable_state=stable_states[env_index],
                is_stable=bool(is_stable[env_index]),
                best_target_distance=int(best_target_distance[env_index]),
                ranking_score=rank_result(
                    float(episode_scores[env_index]),
                    bool(is_stable[env_index]),
                    final_states[env_index],
                    target_x,
                    target_y,
                    int(initial_lives[env_index]),
                ),
            )
            results.append(result)

    return results, len(candidates), total_env_steps


def expand_frontier(
    frontier: list[Candidate],
    macro_library: tuple[Macro, ...],
) -> list[Candidate]:
    return [
        Candidate(sequence=candidate.sequence + (macro,))
        for candidate in frontier
        for macro in macro_library
    ]


def select_new_frontier(
    results: list[CandidateResult],
    visited: set[tuple[int, ...]],
    frontier_limit: int,
) -> tuple[list[CandidateResult], list[CandidateResult]]:
    unique_new: dict[tuple[int, ...], CandidateResult] = {}
    scored_or_room_change: list[CandidateResult] = []
    for result in sorted(
        results,
        key=lambda item: (
            item.score,
            item.final_state.room_id != 1,
            item.is_stable,
            item.ranking_score,
            -item.best_target_distance,
            -item.episode_length,
        ),
        reverse=True,
    ):
        if result.score > 0.0 or result.final_state.room_id != 1:
            scored_or_room_change.append(result)
        if not result.is_stable:
            continue
        key = state_key(result.stable_state)
        if key in visited or key in unique_new:
            continue
        unique_new[key] = result
        if len(unique_new) >= frontier_limit:
            break
    return list(unique_new.values()), scored_or_room_change


def default_macro_library() -> str:
    actions = [
        "NOOP",
        "UP",
        "DOWN",
        "LEFT",
        "RIGHT",
        "UPLEFT",
        "UPRIGHT",
        "DOWNLEFT",
        "DOWNRIGHT",
        "LEFTFIRE",
        "RIGHTFIRE",
        "UPLEFTFIRE",
        "UPRIGHTFIRE",
        "DOWNLEFTFIRE",
        "DOWNRIGHTFIRE",
    ]
    steps = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32]
    return ",".join(f"{name}:{count}" for name in actions for count in steps)


def run_state_graph_search(args: argparse.Namespace) -> None:
    if args.noop_max <= 0:
        raise ValueError(
            "--noop-max must be >= 1 for the current Atari env reset path."
        )

    frontier = [Candidate(sequence=())]
    visited: set[tuple[int, ...]] = set()
    macro_library = parse_macro_library(args.macro_library)
    log_path = Path(args.log_path)
    summary_path = Path(args.summary_path)

    best_score = float("-inf")
    best_result: CandidateResult | None = None
    for depth in range(1, args.depth + 1):
        candidates = expand_frontier(frontier, macro_library)
        results, num_candidates, env_steps = evaluate_candidates(
            candidates=candidates,
            batch_size=args.batch_size,
            frame_skip=args.frame_skip,
            noop_max=args.noop_max,
            seed=args.seed,
            settle_steps=args.settle_steps,
            stable_window=args.stable_window,
            target_x=args.target_x,
            target_y=args.target_y,
        )
        new_frontier, scored_results = select_new_frontier(
            results,
            visited,
            args.frontier_limit,
        )
        for result in new_frontier:
            visited.add(state_key(result.stable_state))
        frontier = [result.candidate for result in new_frontier]

        depth_best = max(
            results,
            key=lambda item: (
                item.score,
                item.final_state.room_id != 1,
                item.is_stable,
                item.ranking_score,
                -item.best_target_distance,
            ),
        )
        if best_result is None or (
            depth_best.score,
            depth_best.final_state.room_id != 1,
            depth_best.is_stable,
            depth_best.ranking_score,
            -depth_best.best_target_distance,
        ) > (
            best_result.score,
            best_result.final_state.room_id != 1,
            best_result.is_stable,
            best_result.ranking_score,
            -best_result.best_target_distance,
        ):
            best_result = depth_best
        best_score = max(best_score, depth_best.score)

        print(
            "depth_summary:",
            f"depth={depth}",
            f"num_candidates={num_candidates}",
            f"env_steps={env_steps}",
            f"new_states={len(new_frontier)}",
            f"frontier_size={len(frontier)}",
            f"visited_states={len(visited)}",
            f"score_max={depth_best.score:.1f}",
            f"best_target_distance={depth_best.best_target_distance}",
            f"best_is_stable={depth_best.is_stable}",
            f"best_final_state={state_to_dict(depth_best.final_state)}",
            f"best_stable_state={state_to_dict(depth_best.stable_state)}",
            f"best_sequence={sequence_to_repr(depth_best.candidate.sequence)}",
        )
        if scored_results:
            print(
                "scored_or_room_change:",
                [
                    {
                        "score": result.score,
                        "final_state": state_to_dict(result.final_state),
                        "sequence": sequence_to_repr(result.candidate.sequence),
                    }
                    for result in scored_results[:8]
                ],
            )

        append_trial_record(
            log_path,
            {
                "timestamp": datetime.now()
                .astimezone()
                .isoformat(timespec="seconds"),
                "trial_name": f"{args.trial_name}_depth{depth:02d}",
                "kind": "state_graph_search",
                "game": "MontezumaRevenge-v5",
                "depth": depth,
                "num_candidates": num_candidates,
                "episodes_started": num_candidates,
                "episodes_finished": num_candidates,
                "env_steps": env_steps,
                "ale_frames": env_steps * args.frame_skip,
                "new_states": len(new_frontier),
                "frontier_size": len(frontier),
                "visited_states": len(visited),
                "score_max": depth_best.score,
                "best_score_so_far": best_score,
                "best_target_distance": depth_best.best_target_distance,
                "best_sequence": sequence_to_repr(
                    depth_best.candidate.sequence
                ),
                "best_final_state": state_to_dict(depth_best.final_state),
                "best_stable_state": state_to_dict(depth_best.stable_state),
                "best_is_stable": depth_best.is_stable,
                "new_frontier": [
                    {
                        "sequence": sequence_to_repr(result.candidate.sequence),
                        "score": result.score,
                        "episode_length": result.episode_length,
                        "ranking_score": result.ranking_score,
                        "best_target_distance": result.best_target_distance,
                        "stable_state": state_to_dict(result.stable_state),
                        "final_state": state_to_dict(result.final_state),
                    }
                    for result in new_frontier
                ],
                "scored_or_room_change": [
                    {
                        "sequence": sequence_to_repr(result.candidate.sequence),
                        "score": result.score,
                        "episode_length": result.episode_length,
                        "ranking_score": result.ranking_score,
                        "best_target_distance": result.best_target_distance,
                        "stable_state": state_to_dict(result.stable_state),
                        "final_state": state_to_dict(result.final_state),
                    }
                    for result in scored_results[:32]
                ],
                "macro_library": sequence_to_repr(macro_library),
                "frontier_limit": args.frontier_limit,
                "seed": args.seed,
                "frame_skip": args.frame_skip,
                "noop_max": args.noop_max,
                "settle_steps": args.settle_steps,
                "stable_window": args.stable_window,
                "target_x": args.target_x,
                "target_y": args.target_y,
                "notes": args.notes,
            },
        )

        print_summary(write_summary(log_path, summary_path))
        if not frontier:
            print(f"search_stop: depth={depth} reason=no_new_stable_states")
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover a graph of stable Montezuma macro states.",
    )
    parser.add_argument("--trial-name", default="state_graph")
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--frontier-limit", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--noop-max", type=int, default=1)
    parser.add_argument("--settle-steps", type=int, default=32)
    parser.add_argument("--stable-window", type=int, default=4)
    parser.add_argument("--target-x", type=int, default=20)
    parser.add_argument("--target-y", type=int, default=104)
    parser.add_argument("--macro-library", default=default_macro_library())
    parser.add_argument("--log-path", default=str(DEFAULT_LOG_PATH))
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--notes", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run_state_graph_search(parse_args())
