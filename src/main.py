import argparse
import os
import sys
from dotenv import load_dotenv
import asyncio
import contextlib

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

load_dotenv(override=True)

from src.interview_session.interview_session import InterviewSession
from src.utils.speech.speech_to_text import PYAUDIO_AVAILABLE

async def run_terminal_mode(args):
    if args.restart:
        os.system(f"rm -rf {os.getenv('LOGS_DIR')}/{args.user_id}")
        os.system(f"rm -rf {os.getenv('DATA_DIR')}/{args.user_id}")
        print(f"Cleared data for user {args.user_id}")
    
    # Check if voice features are available when requested
    if (args.voice_input or args.voice_output) and not PYAUDIO_AVAILABLE:
        print("\nWarning: Voice features were requested but PyAudio is not installed.")
        print("Continuing without voice features...\n")
        args.voice_input = False
        args.voice_output = False
    
    # Select topic plan and description based on session type
    session_type = args.session_type
    if session_type == "weekly":
        interview_plan_path = os.getenv('INTERVIEW_PLAN_PATH_WEEKLY',
                                        'data/configs/topics_weekly.json')
        interview_description = "Weekly work check-in: tracking how your tasks and work are evolving"
    else:
        interview_plan_path = os.getenv('INTERVIEW_PLAN_PATH_INTAKE',
                                        'data/configs/topics_intake.json')
        interview_description = os.getenv('INTERVIEW_DESCRIPTION',
                                          "Initial intake interview: understanding your role, tasks, and work patterns")

    interview_session = InterviewSession(
        interaction_mode='agent' if args.user_agent else 'terminal',
        user_config={
            "user_id": args.user_id,
            "enable_voice": args.voice_input,
            "restart": args.restart
        },
        interview_config={
            "enable_voice": args.voice_output,
            "interview_description": interview_description,
            "interview_plan_path": interview_plan_path,
            "interview_evaluation": os.getenv('COMPLETION_METRIC'),
            "additional_context_path": args.additional_context_path,
            "initial_user_portrait_path": os.getenv('USER_PORTRAIT_PATH'),
            "session_type": session_type,
        },
        max_turns=args.max_turns
    )
    
    with contextlib.suppress(KeyboardInterrupt):
        await interview_session.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run interviewer with specific user and session')

    # Three modes: terminal, server, and setup_db
    parser.add_argument('--mode', default='terminal')
    
    # Terminal mode arguments
    parser.add_argument('--user_id', help='User ID for the session')
    parser.add_argument('--additional_context_path', help='Load additional context from a file', default=None)
    parser.add_argument('--user_agent', action='store_true', default=False, 
                        help='Use user agent')
    parser.add_argument('--voice_output', action='store_true', default=False, 
                        help='Enable voice output')
    parser.add_argument('--voice_input', action='store_true', default=False, 
                        help='Enable voice input')
    parser.add_argument('--restart', action='store_true', default=False, 
                        help='Restart the session')
    parser.add_argument('--max_turns', type=int, default=None,
                        help='Maximum number of turns before ending the session')
    parser.add_argument('--session_type', default='intake',
                        choices=['intake', 'weekly'],
                        help='Session type: "intake" for initial profiling (default), "weekly" for recurring check-ins')
    args = parser.parse_args()
    
    # Run the appropriate mode
    if args.mode == 'terminal':
        if not args.user_id:
            parser.error("--user_id is required for terminal mode")
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(run_terminal_mode(args))
    else:
        parser.error(f"Invalid mode: {args.mode}")
    
