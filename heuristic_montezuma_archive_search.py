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

"""Archive/beam search for hand-written Montezuma macro policies.

The search is still fully heuristic: every candidate is a short sequence of
discrete Atari actions repeated for a fixed number of frames. After each depth
expansion, candidates are ranked by actual score first, then by whether they
reach a new room, then by how close they ever get to the first-room key area.

Each depth expansion appends one JSONL record to
`heuristic_montezuma_archive_trials.jsonl` and rewrites
`heuristic_montezuma_archive_trials_summary.csv` with cumulative step counts so
the sample-efficiency curve can be plotted directly.
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
DEFAULT_LOG_PATH = SCRIPT_DIR / "heuristic_montezuma_archive_trials.jsonl"
DEFAULT_SUMMARY_PATH = (
    SCRIPT_DIR / "heuristic_montezuma_archive_trials_summary.csv"
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
    best_key_distance: int
    best_key_state: MontezumaState
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
    lives = int(np.asarray(info["lives"])[env_index])
    return MontezumaState(
        player_x=int(ram[42]),
        player_y=(312 - int(ram[43])) & 255,
        motion_state=int(ram[2]),
        platform_state=int(ram[89]),
        skull_x=int(ram[47]),
        room_id=int(ram[3]),
        lives=lives,
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
                "beam_width": record.get("beam_width", ""),
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
                "best_ranking_score": record.get("best_ranking_score", ""),
                "best_sequence": record.get("best_sequence", ""),
            }
        )

    fieldnames = [
        "trial_index",
        "timestamp",
        "trial_name",
        "depth",
        "num_candidates",
        "beam_width",
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
        "best_ranking_score",
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
    best_row = max(
        rows,
        key=lambda row: (
            float(row["best_score_so_far"]),
            -999999
            if row["best_key_distance_so_far"] in ("", None)
            else -int(row["best_key_distance_so_far"]),
        ),
    )
    last_row = rows[-1]
    print(
        "trial_log_summary:",
        f"records={len(rows)}",
        f"cumulative_env_steps={last_row['cumulative_env_steps']}",
        f"cumulative_ale_frames={last_row['cumulative_ale_frames']}",
    )
    print(
        "best_progress:",
        f"trial_index={best_row['trial_index']}",
        f"trial_name={best_row['trial_name']}",
        f"best_score_so_far={best_row['best_score_so_far']:.1f}",
        f"best_key_distance_so_far={best_row['best_key_distance_so_far']}",
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


def sequence_to_repr(sequence: tuple[Macro, ...]) -> list[list[Any]]:
    return [[MontezumaAction(macro.action).name, macro.steps] for macro in sequence]


def candidate_key(result: CandidateResult) -> tuple[int, ...]:
    final_state = result.final_state
    return (
        final_state.room_id,
        final_state.lives,
        final_state.motion_state,
        final_state.platform_state,
        final_state.player_x,
        final_state.player_y,
        final_state.skull_x // 2,
        int(result.score),
        result.best_key_distance,
    )


def rank_candidate(
    score: float,
    final_state: MontezumaState,
    best_key_distance: int,
    initial_lives: int,
    ranking_mode: str,
    target_x: int,
    target_y: int,
) -> float:
    rank = score * 1_000_000.0
    rank += 50_000.0 * float(final_state.room_id != 1)
    if ranking_mode == "key":
        if final_state.lives == initial_lives:
            rank += 2000.0
        if final_state.motion_state == 4 and final_state.platform_state != 0:
            rank += 500.0
        if final_state.lives == initial_lives:
            rank += 10.0 * float(max(0, 200 - best_key_distance))
            rank -= 2.0 * abs(float(final_state.player_x - target_x))
            rank -= abs(float(final_state.player_y - target_y))
        else:
            rank -= 5000.0 * float(initial_lives - final_state.lives)
    elif ranking_mode == "bottom_left":
        if final_state.lives == initial_lives:
            rank += 2000.0
            rank += 10.0 * float(final_state.player_y)
            rank -= 4.0 * abs(float(final_state.player_x - target_x))
            rank -= 2.0 * abs(float(final_state.player_y - target_y))
        else:
            rank -= 5000.0
        if final_state.motion_state == 4 and final_state.platform_state != 0:
            rank += 50.0
        elif final_state.motion_state == 7 and final_state.platform_state == 0:
            rank += 25.0
    else:
        raise ValueError(f"Unknown ranking_mode={ranking_mode!r}")
    return rank


def evaluate_candidates(
    candidates: list[Candidate],
    batch_size: int,
    frame_skip: int,
    noop_max: int,
    seed: int,
    img_height: int,
    img_width: int,
    gray_scale: bool,
    settle_steps: int,
    stable_window: int,
    key_x: int,
    key_y: int,
    ranking_mode: str,
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
            img_height=img_height,
            img_width=img_width,
            gray_scale=gray_scale,
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
        best_key_distance = np.full(num_envs, 999999, dtype=np.int64)
        best_key_states = [decode_state(info, i) for i in range(num_envs)]
        prev_states = list(best_key_states)
        final_states = list(best_key_states)
        stable_noop_run = np.zeros(num_envs, dtype=np.int64)

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
                prev_state = prev_states[env_index]
                if state.lives == int(initial_lives[env_index]):
                    distance = abs(state.player_x - key_x) + abs(
                        state.player_y - key_y
                    )
                    if distance < best_key_distance[env_index]:
                        best_key_distance[env_index] = distance
                        best_key_states[env_index] = state
                if (
                    actions[env_index] == int(MontezumaAction.NOOP)
                    and state.lives == prev_state.lives
                    and state.player_x == prev_state.player_x
                    and state.player_y == prev_state.player_y
                    and state.motion_state == 4
                ):
                    stable_noop_run[env_index] += 1
                else:
                    stable_noop_run[env_index] = 0
                prev_states[env_index] = state
                if bool(step_done[env_index]) or t + 1 >= horizons[env_index]:
                    done[env_index] = True
                    final_states[env_index] = state

            if np.all(done):
                break

        total_env_steps += int(episode_lengths.sum())
        for env_index, candidate in enumerate(chunk):
            final_state = final_states[env_index]
            result = CandidateResult(
                candidate=candidate,
                score=float(episode_scores[env_index]),
                episode_length=int(episode_lengths[env_index]),
                final_state=final_state,
                best_key_distance=int(best_key_distance[env_index]),
                best_key_state=best_key_states[env_index],
                ranking_score=rank_candidate(
                    float(episode_scores[env_index]),
                    final_state,
                    int(best_key_distance[env_index]),
                    int(initial_lives[env_index]),
                    ranking_mode,
                    key_x,
                    key_y,
                ),
            )
            results.append(result)

    return results, len(candidates), total_env_steps


def select_beam(
    results: list[CandidateResult],
    beam_width: int,
    x_bucket_size: int,
    y_bucket_size: int,
    per_bucket_limit: int,
) -> list[CandidateResult]:
    deduped: dict[tuple[int, ...], CandidateResult] = {}
    bucket_counts: dict[tuple[int, ...], int] = {}
    overflow: list[CandidateResult] = []
    for result in sorted(
        results,
        key=lambda item: (
            item.ranking_score,
            item.score,
            -item.best_key_distance,
            -item.episode_length,
        ),
        reverse=True,
    ):
        key = candidate_key(result)
        if key in deduped:
            continue
        bucket = (
            result.final_state.room_id,
            result.final_state.lives,
            result.final_state.motion_state,
            result.final_state.player_x // max(x_bucket_size, 1),
            result.final_state.player_y // max(y_bucket_size, 1),
        )
        if (
            per_bucket_limit > 0
            and bucket_counts.get(bucket, 0) >= per_bucket_limit
        ):
            overflow.append(result)
            continue
        deduped[key] = result
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        if len(deduped) >= beam_width:
            break
    if len(deduped) < beam_width:
        for result in overflow:
            key = candidate_key(result)
            if key not in deduped:
                deduped[key] = result
            if len(deduped) >= beam_width:
                break
    return list(deduped.values())


def expand_beam(
    beam: list[Candidate],
    macro_library: tuple[Macro, ...],
) -> list[Candidate]:
    return [
        Candidate(sequence=candidate.sequence + (macro,))
        for candidate in beam
        for macro in macro_library
    ]


def run_archive_search(args: argparse.Namespace) -> None:
    if args.noop_max <= 0:
        raise ValueError(
            "--noop-max must be >= 1 for the current Atari env reset path."
        )

    macro_library = parse_macro_library(args.macro_library)
    prefix_sequence = ()
    if args.prefix_sequence.strip():
        prefix_sequence = parse_macro_library(args.prefix_sequence)
    frontier = [Candidate(sequence=prefix_sequence)]
    log_path = Path(args.log_path)
    summary_path = Path(args.summary_path)

    for depth in range(1, args.depth + 1):
        candidates = expand_beam(frontier, macro_library)
        results, num_candidates, env_steps = evaluate_candidates(
            candidates=candidates,
            batch_size=args.batch_size,
            frame_skip=args.frame_skip,
            noop_max=args.noop_max,
            seed=args.seed,
            img_height=args.img_height,
            img_width=args.img_width,
            gray_scale=args.gray_scale,
            settle_steps=args.settle_steps,
            stable_window=args.stable_window,
            key_x=args.key_x,
            key_y=args.key_y,
            ranking_mode=args.ranking_mode,
        )
        selected = select_beam(
            results,
            args.beam_width,
            args.x_bucket_size,
            args.y_bucket_size,
            args.per_bucket_limit,
        )
        frontier = [result.candidate for result in selected]
        best_result = max(
            results,
            key=lambda item: (
                item.score,
                item.final_state.room_id != 1,
                item.ranking_score,
                -item.best_key_distance,
            ),
        )

        print(
            "depth_summary:",
            f"depth={depth}",
            f"num_candidates={num_candidates}",
            f"env_steps={env_steps}",
            f"score_max={best_result.score:.1f}",
            f"best_key_distance={best_result.best_key_distance}",
            f"best_final_state={state_to_dict(best_result.final_state)}",
            f"best_key_state={state_to_dict(best_result.best_key_state)}",
            f"best_sequence={sequence_to_repr(best_result.candidate.sequence)}",
        )

        append_trial_record(
            log_path,
            {
                "timestamp": datetime.now()
                .astimezone()
                .isoformat(timespec="seconds"),
                "trial_name": f"{args.trial_name}_depth{depth:02d}",
                "kind": "archive_search",
                "game": "MontezumaRevenge-v5",
                "depth": depth,
                "num_candidates": num_candidates,
                "beam_width": args.beam_width,
                "episodes_started": num_candidates,
                "episodes_finished": num_candidates,
                "env_steps": env_steps,
                "ale_frames": env_steps * args.frame_skip,
                "score_max": best_result.score,
                "best_room_id": best_result.final_state.room_id,
                "best_key_distance": best_result.best_key_distance,
                "best_ranking_score": best_result.ranking_score,
                "best_sequence": sequence_to_repr(
                    best_result.candidate.sequence
                ),
                "prefix_sequence": sequence_to_repr(prefix_sequence),
                "best_final_state": state_to_dict(best_result.final_state),
                "best_key_state": state_to_dict(best_result.best_key_state),
                "selected_frontier": [
                    {
                        "sequence": sequence_to_repr(
                            result.candidate.sequence
                        ),
                        "score": result.score,
                        "episode_length": result.episode_length,
                        "ranking_score": result.ranking_score,
                        "best_key_distance": result.best_key_distance,
                        "final_state": state_to_dict(result.final_state),
                        "best_key_state": state_to_dict(result.best_key_state),
                    }
                    for result in selected
                ],
                "macro_library": sequence_to_repr(macro_library),
                "seed": args.seed,
                "frame_skip": args.frame_skip,
                "noop_max": args.noop_max,
                "img_height": args.img_height,
                "img_width": args.img_width,
                "gray_scale": args.gray_scale,
                "settle_steps": args.settle_steps,
                "stable_window": args.stable_window,
                "key_x": args.key_x,
                "key_y": args.key_y,
                "ranking_mode": args.ranking_mode,
                "x_bucket_size": args.x_bucket_size,
                "y_bucket_size": args.y_bucket_size,
                "per_bucket_limit": args.per_bucket_limit,
                "notes": args.notes,
            },
        )

        print_summary(write_summary(log_path, summary_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Beam/archive search over deterministic Montezuma macros.",
    )
    parser.add_argument("--trial-name", default="archive_beam")
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--noop-max", type=int, default=1)
    parser.add_argument("--img-height", type=int, default=1)
    parser.add_argument("--img-width", type=int, default=1)
    parser.add_argument(
        "--gray-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--settle-steps", type=int, default=8)
    parser.add_argument("--stable-window", type=int, default=4)
    parser.add_argument("--prefix-sequence", default="")
    parser.add_argument("--key-x", type=int, default=20)
    parser.add_argument("--key-y", type=int, default=104)
    parser.add_argument("--x-bucket-size", type=int, default=8)
    parser.add_argument("--y-bucket-size", type=int, default=8)
    parser.add_argument("--per-bucket-limit", type=int, default=4)
    parser.add_argument(
        "--ranking-mode",
        choices=("key", "bottom_left"),
        default="key",
    )
    parser.add_argument(
        "--macro-library",
        default=(
            "NOOP:4,NOOP:8,NOOP:16,"
            "UP:2,UP:4,UP:8,UP:16,"
            "DOWN:2,DOWN:4,DOWN:8,DOWN:16,DOWN:24,"
            "LEFT:2,LEFT:4,LEFT:8,LEFT:16,"
            "RIGHT:2,RIGHT:4,RIGHT:8,RIGHT:16,"
            "LEFTFIRE:1,LEFTFIRE:2,LEFTFIRE:4,LEFTFIRE:8,LEFTFIRE:16,"
            "RIGHTFIRE:1,RIGHTFIRE:2,RIGHTFIRE:4,RIGHTFIRE:8,RIGHTFIRE:16,"
            "UPLEFTFIRE:2,UPLEFTFIRE:4,UPLEFTFIRE:8,"
            "UPRIGHTFIRE:2,UPRIGHTFIRE:4,UPRIGHTFIRE:8,"
            "DOWNLEFTFIRE:2,DOWNLEFTFIRE:4,DOWNLEFTFIRE:8,"
            "DOWNRIGHTFIRE:2,DOWNRIGHTFIRE:4,DOWNRIGHTFIRE:8"
        ),
    )
    parser.add_argument("--log-path", default=str(DEFAULT_LOG_PATH))
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--notes", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run_archive_search(parse_args())
