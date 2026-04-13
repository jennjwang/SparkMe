"""
run_simulation.py — Run a SparkMe interview with a user simulator.

Creates a fresh synthetic user (copying profile from an existing user) so
each simulation starts with a clean slate. The synthetic user_id is written
to stdout so downstream eval scripts can consume it.

Usage
-----
# Single run, cooperative user
python scripts/run_simulation.py --profile-user-id 1T8lGuWK6w-0q4S-s2_KeA --max-turns 30

# With hesitancy
USER_AGENT_HESITANCY=0.5 python scripts/run_simulation.py \
    --profile-user-id 1T8lGuWK6w-0q4S-s2_KeA --max-turns 30

# Batch: run multiple profiles / hesitancy levels
python scripts/run_simulation.py \
    --profile-user-id 1T8lGuWK6w-0q4S-s2_KeA WF31MF3_WtHzW78V6hH2lg \
    --hesitancy 0.0 0.5 \
    --max-turns 30

# Skip creating a new sim user (reuse an existing sim user_id)
python scripts/run_simulation.py --sim-user-id sim_abc123 --max-turns 30
"""

import argparse
import asyncio
import contextlib
import json
import os
import shutil
import sys
import time

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from dotenv import load_dotenv
load_dotenv(override=True)


def create_sim_user(profile_user_id: str, profiles_dir: str, sim_user_id: str) -> str:
    """Copy profile files from profile_user_id to sim_user_id directory."""
    src_dir = os.path.join(profiles_dir, profile_user_id)
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"Profile directory not found: {src_dir}")

    dst_dir = os.path.join(profiles_dir, sim_user_id)
    os.makedirs(dst_dir, exist_ok=True)

    for fname in os.listdir(src_dir):
        src_file = os.path.join(src_dir, fname)
        # Rename files that embed the original user_id
        dst_fname = fname.replace(profile_user_id, sim_user_id)
        dst_file = os.path.join(dst_dir, dst_fname)
        shutil.copy2(src_file, dst_file)

    print(f"[sim] Created profile for {sim_user_id} (from {profile_user_id})")
    return sim_user_id


async def run_session(sim_user_id: str, max_turns: int, session_type: str):
    from src.interview_session.interview_session import InterviewSession

    if session_type == "weekly":
        interview_plan_path = os.getenv("INTERVIEW_PLAN_PATH_WEEKLY", "configs/topics_weekly.json")
        interview_description = "Weekly work check-in"
    else:
        interview_plan_path = os.getenv("INTERVIEW_PLAN_PATH_INTAKE", "configs/topics_intake.json")
        interview_description = os.getenv(
            "INTERVIEW_DESCRIPTION",
            "Initial intake interview: understanding your role, tasks, and work patterns"
        )

    session = InterviewSession(
        interaction_mode="agent",
        user_config={"user_id": sim_user_id, "enable_voice": False, "restart": False},
        interview_config={
            "enable_voice": False,
            "interview_description": interview_description,
            "interview_plan_path": interview_plan_path,
            "interview_evaluation": os.getenv("COMPLETION_METRIC"),
            "additional_context_path": None,
            "initial_user_portrait_path": os.getenv("USER_PORTRAIT_PATH"),
            "session_type": session_type,
        },
        max_turns=max_turns,
    )
    with contextlib.suppress(KeyboardInterrupt):
        await session.run()


def main():
    parser = argparse.ArgumentParser(description="Run SparkMe with a user simulator")
    parser.add_argument("--profile-user-id", nargs="+",
                        help="Source profile user_id(s) to simulate")
    parser.add_argument("--sim-user-id", nargs="*",
                        help="Override synthetic user_id(s) (skip profile copy if set)")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--hesitancy", nargs="*", type=float, default=[0.0],
                        help="Hesitancy level(s) 0.0–1.0 (default 0.0)")
    parser.add_argument("--session-type", default="intake", choices=["intake", "weekly"])
    parser.add_argument("--profiles-dir", default=None,
                        help="Override USER_AGENT_PROFILES_DIR")
    args = parser.parse_args()

    profiles_dir = args.profiles_dir or os.getenv("USER_AGENT_PROFILES_DIR", "data/sample_user_profiles")

    # Build the list of (sim_user_id, profile_user_id) pairs to run
    runs = []

    if args.sim_user_id:
        for sim_uid in args.sim_user_id:
            for h in args.hesitancy:
                runs.append((sim_uid, None, h))
    elif args.profile_user_id:
        for profile_uid in args.profile_user_id:
            for h in args.hesitancy:
                tag = f"h{int(h*100):03d}"
                ts = int(time.time())
                sim_uid = f"sim_{profile_uid[:8]}_{tag}_{ts}"
                runs.append((sim_uid, profile_uid, h))
    else:
        parser.error("Provide --profile-user-id or --sim-user-id")

    sim_user_ids = []
    for sim_uid, profile_uid, hesitancy in runs:
        if profile_uid:
            create_sim_user(profile_uid, profiles_dir, sim_uid)

        os.environ["USER_AGENT_HESITANCY"] = str(hesitancy)
        print(f"\n[sim] Running: user={sim_uid}  hesitancy={hesitancy}  max_turns={args.max_turns}")
        asyncio.run(run_session(sim_uid, args.max_turns, args.session_type))
        sim_user_ids.append(sim_uid)
        print(f"[sim] Done: {sim_uid}")

    # Write out the sim user IDs so eval scripts can pick them up
    out = [{"User ID": uid} for uid in sim_user_ids]
    out_path = "analysis/sample_users_sim.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[sim] Wrote {len(sim_user_ids)} sim user(s) to {out_path}")


if __name__ == "__main__":
    main()
