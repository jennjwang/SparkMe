"""
Flask Web Application for Interview Session
Supports both text and voice input/output with authentication
"""

from flask import Flask, request, jsonify, render_template, Response, redirect, url_for, flash, stream_with_context
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from functools import wraps
import traceback
import asyncio
import concurrent.futures
import threading
from collections import OrderedDict
import os
import sys
import uuid
import argparse
import time
import logging
import secrets
import re
from datetime import datetime, timezone
# import hashlib  # unused when name-only login is active
import json
from logging.handlers import RotatingFileHandler
from pathlib import Path


class _SuppressPollingEndpoints(logging.Filter):
    _SUPPRESS = {'/api/session-state', '/api/get-messages'}

    def filter(self, record):
        msg = record.getMessage()
        return not any(ep in msg for ep in self._SUPPRESS)


logging.getLogger('werkzeug').addFilter(_SuppressPollingEndpoints())
from typing import Dict, Optional
from dotenv import load_dotenv

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

load_dotenv(override=True)

# Your backend imports
from src.utils.speech.text_to_speech import TextToSpeechBase, create_tts_engine
from src.utils.speech.audio_player import AudioPlayerBase, create_audio_player
from src.utils.speech.speech_to_text import create_stt_engine
from src.utils.logger.evaluation_logger import EvaluationLogger
from src.interview_session.interview_session import InterviewSession
from src.content.session_agenda.session_agenda import normalize_user_portrait

# =============================================================================
# CONFIGURATION
# =============================================================================

SESSION_TIMEOUT_SECONDS = 3600  # 1 hour
START_TIME = time.time()
DEFAULT_AVAILABLE_TIME_MINUTES = 10
MIN_AVAILABLE_TIME_MINUTES = 5
MAX_AVAILABLE_TIME_MINUTES = 120

class AppConfig:
    """Application configuration"""
    def __init__(self):
        self.default_user_id = "web_user"
        self.host = "0.0.0.0"
        self.port = 5000
        self.debug = False
        self.restart = False
        self.max_turns = None
        self.additional_context_path = None

config = AppConfig()

# TTS/STT engines
TTS_PROVIDER = os.getenv('TTS_PROVIDER', 'openai')
TTS_VOICE = os.getenv('TTS_VOICE', 'alloy')
tts_engine: TextToSpeechBase = create_tts_engine(provider=TTS_PROVIDER, voice=TTS_VOICE)
stt_engine = create_stt_engine()

# =============================================================================
# FLASK APP SETUP
# =============================================================================

app = Flask(__name__,
            static_folder='web/static',
            template_folder='web/templates')

app.secret_key = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(32))
CORS(app)

# =============================================================================
# AUTHENTICATION SETUP
# =============================================================================

REQUIRE_LOGIN = os.getenv('REQUIRE_LOGIN', 'true').lower() == 'true'
LOGIN_ALWAYS_NEW_USER_ID = os.getenv('LOGIN_ALWAYS_NEW_USER_ID', 'true').lower() == 'true'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = None

USERS_FILE = os.path.join(os.getenv('DATA_DIR', 'data'), 'users.json')

class User(UserMixin):
    def __init__(self, user_id, username):
        self.id = user_id
        self.username = username

def load_users():
    """Load users from JSON file"""
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}


def _generate_random_numeric_string(digits: int = 8) -> str:
    """Generate a fixed-length random numeric string."""
    digits = max(1, int(digits))
    if digits == 1:
        return str(secrets.randbelow(10))
    lower = 10 ** (digits - 1)
    span = 9 * lower
    return str(lower + secrets.randbelow(span))


def _generate_unique_user_id(users: dict, digits: int = 10) -> str:
    """Generate a numeric user_id not present in users."""
    for _ in range(64):
        candidate = _generate_random_numeric_string(digits)
        if candidate not in users:
            return candidate
    return secrets.token_urlsafe(16)


def _safe_user_id_from_prolific_pid(prolific_pid: str) -> str:
    """Use the Prolific PID as the app user id, normalized for filesystem paths."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(prolific_pid or "").strip())
    return safe.strip("._-") or _generate_random_numeric_string(10)

def save_users(users):
    """Save users to JSON file"""
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

# def hash_password(password):
#     """Hash password using SHA-256"""
#     return hashlib.sha256(password.encode()).hexdigest()

_ANON_USER_ID = os.getenv('ANON_USER_ID', 'anon')
_ANON_USERNAME = os.getenv('ANON_USERNAME', 'anon')

if not REQUIRE_LOGIN:
    # Override Flask-Login's login_required to be a no-op
    def login_required(f):
        return f

def get_current_user():
    """Return current_user if login is required, otherwise return a fixed anon user."""
    if REQUIRE_LOGIN:
        return current_user
    if current_user.is_authenticated:
        return current_user
    return User(_ANON_USER_ID, _ANON_USERNAME)

@login_manager.user_loader
def load_user(user_id):
    users = load_users()
    if user_id in users:
        return User(user_id, users[user_id]['username'])
    return None

# =============================================================================
# LOGGING SETUP
# =============================================================================

if not app.debug:
    os.makedirs('logs', exist_ok=True)
    file_handler = RotatingFileHandler(
        'logs/flask_app.log',
        maxBytes=10485760,  # 10MB
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Interview application startup')

# =============================================================================
# ASYNC EVENT LOOP MANAGEMENT
# =============================================================================

loop = asyncio.new_event_loop()

def start_background_loop(loop):
    """Run async event loop in background thread."""
    asyncio.set_event_loop(loop)
    loop.run_forever()

threading.Thread(target=start_background_loop, args=(loop,), daemon=True).start()

def run_async_task(coro):
    """Submit coroutine to background loop."""
    return asyncio.run_coroutine_threadsafe(coro, loop)

# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

class SessionWrapper:
    def __init__(self, session_token: str, interview_session: InterviewSession,
                 user_id: str):
        self.session_token = session_token
        self.interview_session = interview_session
        self.user_id = user_id
        self.created_at = time.time()

active_sessions: Dict[str, SessionWrapper] = {}
session_audio_cache: Dict[str, Dict[str, object]] = {}
# Rolling dialogue state for /api/task-followup, keyed by session + phase.
task_followup_history_by_session: Dict[str, Dict[str, list[Dict[str, str]]]] = {}
# Tracks follow-up turn count since last task extraction, per session per phase
task_followup_turns_by_session: Dict[str, Dict[str, int]] = {}
# Tracks how many chat_history entries have been delivered for agent-mode sessions
chat_history_offsets: Dict[str, int] = {}
# Pending turn metadata keyed by session token -> turn id.
pending_turns_by_session: Dict[str, OrderedDict[str, Dict[str, object]]] = {}
# Guards against duplicate latency rows when messages are replayed.
delivered_turn_messages_by_session: Dict[str, set[str]] = {}


_TASK_FOLLOWUP_HISTORY_MAX_MESSAGES = 12


def _normalize_task_followup_phase(raw_phase: object) -> str:
    phase = str(raw_phase or "probing").strip().lower()
    return phase if phase in {"probing", "ai_extras", "ai_open"} else "probing"


def _normalize_task_followup_dialogue(raw_dialogue: object) -> list[Dict[str, str]]:
    if not isinstance(raw_dialogue, list):
        return []
    out: list[Dict[str, str]] = []
    for item in raw_dialogue:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        role_raw = str(item.get("role", "")).strip().lower()
        if role_raw in {"user", "participant", "human"}:
            role = "Participant"
        elif role_raw in {"assistant", "interviewer", "bot"}:
            role = "Interviewer"
        else:
            continue
        out.append({"role": role, "content": content})
    return out[-_TASK_FOLLOWUP_HISTORY_MAX_MESSAGES:]


def _task_followup_history(session_token: str, phase: str) -> list[Dict[str, str]]:
    bucket = task_followup_history_by_session.setdefault(session_token, {})
    return bucket.setdefault(phase, [])


def _format_task_followup_history(history: list[Dict[str, str]]) -> str:
    if not history:
        return "(none)"
    lines = []
    for item in history[-_TASK_FOLLOWUP_HISTORY_MAX_MESSAGES:]:
        role = str(item.get("role", "")).strip() or "Participant"
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(none)"


def _latest_interviewer_message_from_chat(iv) -> str:
    for msg in reversed(getattr(iv, "chat_history", []) or []):
        role = str(getattr(msg, "role", "")).strip().lower()
        mtype = str(getattr(msg, "type", "")).strip().lower()
        content = str(getattr(msg, "content", "")).strip()
        if role == "interviewer" and mtype == "conversation" and content:
            return content
    return ""


def _parse_iso_datetime(value: Optional[object]) -> Optional[datetime]:
    """Parse ISO timestamps into timezone-aware datetimes."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    except Exception:
        return None


def _register_pending_turn(session_token: str, turn_id: str, payload: Dict[str, object]) -> None:
    """Track a user turn until at least one assistant response is delivered."""
    turns = pending_turns_by_session.setdefault(session_token, OrderedDict())
    turns[turn_id] = payload
    # Keep bounded state per session.
    while len(turns) > 500:
        turns.popitem(last=False)


def _normalize_available_time_minutes(raw_value: Optional[object], fallback: Optional[int] = None) -> Optional[int]:
    """Normalize available_time input to a bounded number of minutes."""
    if raw_value is None:
        return None
    try:
        minutes = int(float(raw_value))
    except (TypeError, ValueError):
        return fallback
    return max(MIN_AVAILABLE_TIME_MINUTES, min(MAX_AVAILABLE_TIME_MINUTES, minutes))


def _serialize_session_start_time(value: Optional[datetime]) -> tuple[Optional[str], Optional[int]]:
    """Serialize session start time as UTC ISO string and epoch ms."""
    if value is None:
        return None, None
    try:
        if value.tzinfo is None:
            utc_value = value.astimezone(timezone.utc)
        else:
            utc_value = value.astimezone(timezone.utc)
        return utc_value.isoformat().replace("+00:00", "Z"), int(utc_value.timestamp() * 1000)
    except Exception:
        return None, None

def create_interview_session(user_id: str, session_type: str = "intake",
                             interaction_mode: str = 'api',
                             available_time: int = None) -> tuple[InterviewSession, str]:
    """Create interview session with authenticated user_id.

    Args:
        user_id: User identifier (for agent mode, use the sample profile user_id).
        session_type: "intake" for initial profiling, "weekly" for recurring check-ins.
        interaction_mode: 'api' for human-via-web, 'agent' for simulated user.
        available_time: Participant's available time in minutes (from pre-interview question).
    """
    session_token = str(uuid.uuid4())

    if session_type == "weekly":
        interview_plan_path = os.getenv('INTERVIEW_PLAN_PATH_WEEKLY',
                                        'configs/topics_weekly.json')
        interview_description = "Weekly work check-in: tracking how your tasks and work are evolving"
    else:
        interview_plan_path = os.getenv('INTERVIEW_PLAN_PATH_INTAKE',
                                        'configs/topics_intake.json')
        interview_description = os.getenv(
            'INTERVIEW_DESCRIPTION',
            "Initial intake interview: understanding your role, tasks, and work patterns"
        )

    interview_session = InterviewSession(
        interaction_mode=interaction_mode,
        user_config={
            "user_id": user_id,
            "enable_voice": False,
            "restart": config.restart
        },
        interview_config={
            "enable_voice": False,
            "interview_description": interview_description,
            "interview_plan_path": interview_plan_path,
            "interview_evaluation": os.getenv('COMPLETION_METRIC'),
            "additional_context_path": config.additional_context_path,
            "initial_user_portrait_path": os.getenv('USER_PORTRAIT_PATH'),
            "session_type": session_type,
            "available_time": available_time,
        },
        max_turns=config.max_turns
    )
    wrapper = SessionWrapper(
        session_token=session_token,
        interview_session=interview_session,
        user_id=user_id,
    )
    active_sessions[session_token] = wrapper
    
    session_loop = asyncio.new_event_loop()
    def _start_loop(l):
        asyncio.set_event_loop(l)
        l.run_forever()
    t = threading.Thread(target=_start_loop, args=(session_loop,), daemon=True)
    t.start()

    wrapper.loop = session_loop
    wrapper.loop_thread = t
    asyncio.run_coroutine_threadsafe(interview_session.run(), session_loop)
    
    return interview_session, session_token

def get_session(session_token: str) -> Optional[InterviewSession]:
    wrapper = active_sessions.get(session_token)
    return wrapper.interview_session if wrapper is not None else None

def get_session_wrapper(session_token: str) -> Optional[SessionWrapper]:
    return active_sessions.get(session_token)

def agent_or_login_required(f):
    """Allow access if login is disabled, logged in, or a valid session_token is present (agent mode)."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not REQUIRE_LOGIN:
            return f(*args, **kwargs)
        if current_user.is_authenticated:
            return f(*args, **kwargs)
        token = (request.args.get('session_token') or
                 (request.get_json(silent=True) or {}).get('session_token'))
        if token and token in active_sessions:
            return f(*args, **kwargs)
        return jsonify({'success': False, 'error': 'Authentication required'}), 401
    return decorated

# =============================================================================
# AUTHENTICATION ROUTES (NO @login_required)
# =============================================================================


def _login_user_record(
    username: str,
    *,
    user_id: Optional[str] = None,
    prolific_pid: str = "",
    prolific_study_id: str = "",
    prolific_session_id: str = "",
):
    users = load_users()
    if not user_id:
        user_id = None
        if not LOGIN_ALWAYS_NEW_USER_ID:
            # Legacy mode: reuse an existing account for the same username.
            for uid, user_data in users.items():
                if user_data.get('username') == username:
                    user_id = uid
                    break
        if not user_id:
            user_id = _generate_unique_user_id(users, digits=10)

    existing = users.get(user_id)
    if existing:
        existing['username'] = username
        if prolific_pid:
            existing['prolific_pid'] = prolific_pid
        if prolific_study_id:
            existing['prolific_study_id'] = prolific_study_id
        if prolific_session_id:
            existing['prolific_session_id'] = prolific_session_id
        save_users(users)
        app.logger.info(f"Existing user login: {username} ({user_id})")
    else:
        user_record = {'username': username, 'created_at': time.time()}
        if prolific_pid:
            user_record['prolific_pid'] = prolific_pid
        if prolific_study_id:
            user_record['prolific_study_id'] = prolific_study_id
        if prolific_session_id:
            user_record['prolific_session_id'] = prolific_session_id
        users[user_id] = user_record
        save_users(users)
        os.makedirs(os.path.join(os.getenv('LOGS_DIR', 'logs'), user_id), exist_ok=True)
        os.makedirs(os.path.join(os.getenv('DATA_DIR', 'data'), user_id), exist_ok=True)
        app.logger.info(f"New user created: {username} ({user_id}) prolific_pid={prolific_pid or 'none'}")

    user = User(user_id, username)
    login_user(user)
    app.logger.info(f"User logged in: {username} ({user_id})")
    return user

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if not REQUIRE_LOGIN:
        return redirect(url_for('index'))
    if current_user.is_authenticated:
        return redirect(url_for('index'))  # Changed: redirect to index with instructions

    query_prolific_pid = request.args.get('PROLIFIC_PID', '').strip()

    # Also extract PROLIFIC_PID from the `next` redirect URL (Prolific→/chat→/login flow)
    _next_url = request.args.get('next', '')
    if not query_prolific_pid and _next_url:
        from urllib.parse import urlparse, parse_qs
        _next_qs = parse_qs(urlparse(_next_url).query)
        query_prolific_pid = (_next_qs.get('PROLIFIC_PID') or [''])[0].strip()
        if not query_prolific_pid:
            # handle un-decoded next params
            query_prolific_pid = (_next_qs.get('PROLIFIC_PID%3D') or [''])[0].strip()

    if request.method == 'GET' and query_prolific_pid:
        # Extract study/session IDs from either top-level params or the next URL
        _next_qs_full = {}
        if _next_url:
            from urllib.parse import urlparse, parse_qs
            _next_qs_full = parse_qs(urlparse(_next_url).query)
        study_id = (request.args.get('STUDY_ID', '').strip()
                    or (_next_qs_full.get('STUDY_ID') or [''])[0].strip())
        session_id = (request.args.get('SESSION_ID', '').strip()
                      or (_next_qs_full.get('SESSION_ID') or [''])[0].strip())
        _login_user_record(
            query_prolific_pid,
            user_id=_safe_user_id_from_prolific_pid(query_prolific_pid),
            prolific_pid=query_prolific_pid,
            prolific_study_id=study_id,
            prolific_session_id=session_id,
        )
        return redirect(url_for('index', show_intro='1'))
    
    if request.method == 'POST':
        prolific_pid = request.form.get('prolific_pid', '').strip()
        if prolific_pid:
            username = prolific_pid
            user_id = _safe_user_id_from_prolific_pid(prolific_pid)
        else:
            username = request.form.get('username', '').strip() or _generate_random_numeric_string(8)
            user_id = None

        _login_user_record(
            username,
            user_id=user_id,
            prolific_pid=prolific_pid,
            prolific_study_id=request.form.get('study_id', '').strip(),
            prolific_session_id=request.form.get('prolific_session_id', '').strip(),
        )

        # Always route through the interviewer-info page first.
        return redirect(url_for('index', show_intro='1'))

        # --- password-based login (commented out) ---
        # password = request.form.get('password', '')
        # if user_id and users[user_id]['password'] == hash_password(password):
        #     user = User(user_id, username)
        #     login_user(user)
        #     app.logger.info(f"User logged in: {username} ({user_id})")
        #     next_page = request.args.get('next')
        #     return redirect(next_page if next_page else url_for('index'))
        # else:
        #     flash('Invalid username or password', 'error')

    return render_template(
        'login.html',
        suggested_username=_generate_random_numeric_string(8),
        prolific_pid=query_prolific_pid,
        study_id=request.args.get('STUDY_ID', '').strip(),
        prolific_session_id=request.args.get('SESSION_ID', '').strip(),
    )

# --- /register route (commented out — superseded by name-only login) ---
# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     """Registration page"""
#     if current_user.is_authenticated:
#         return redirect(url_for('index'))
#
#     if request.method == 'POST':
#         username = request.form.get('username', '').strip()
#         password = request.form.get('password', '')
#
#         if not username or not password:
#             flash('Username and password are required', 'error')
#             return render_template('register.html')
#
#         if len(password) < 6:
#             flash('Password must be at least 6 characters', 'error')
#             return render_template('register.html')
#
#         users = load_users()
#
#         for user_data in users.values():
#             if user_data['username'] == username:
#                 flash('Username already exists', 'error')
#                 return render_template('register.html')
#
#         user_id = secrets.token_urlsafe(16)
#         users[user_id] = {
#             'username': username,
#             'password': hash_password(password),
#             'created_at': time.time()
#         }
#         save_users(users)
#         os.makedirs(os.path.join(os.getenv('LOGS_DIR', 'logs'), user_id), exist_ok=True)
#         os.makedirs(os.path.join(os.getenv('DATA_DIR', 'data'), user_id), exist_ok=True)
#         app.logger.info(f"New user registered: {username} ({user_id})")
#         flash('Registration successful! Please login.', 'success')
#         return redirect(url_for('login'))
#
#     return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    """Logout"""
    username = get_current_user().username
    logout_user()
    app.logger.info(f"User logged out: {username}")
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# =============================================================================
# PAGE ROUTES - PROTECTED (REQUIRE LOGIN)
# =============================================================================

@app.route('/')
@login_required  # MUST BE LOGGED IN TO SEE INSTRUCTIONS
def index():
    """Landing page with instructions - shown after login.
    If the user already has a prior session (intake completed), skip to chat
    unless `show_intro=1` is provided."""
    show_intro = request.args.get('show_intro', '').lower() in {'1', 'true', 'yes'}
    logs_dir = os.getenv("LOGS_DIR", "logs")
    user_logs = os.path.join(logs_dir, get_current_user().id, "execution_logs")
    if REQUIRE_LOGIN and not show_intro and os.path.isdir(user_logs):
        session_dirs = [d for d in os.listdir(user_logs)
                        if d.startswith('session_') and
                        os.path.isdir(os.path.join(user_logs, d))]
        if session_dirs:
            return redirect(url_for('unified_chat'))
    return render_template(
        'index.html',
        username=get_current_user().username,
        require_login=REQUIRE_LOGIN,
    )

@app.route('/chat')
@login_required  # MUST BE LOGGED IN
def unified_chat():
    """Unified chat interface"""
    return render_template(
        'chat.html',
        username=get_current_user().username,
        prolific_completion_code=os.getenv('PROLIFIC_COMPLETION_CODE', ''),
        attn_check_max_fails=int(os.getenv('ATTN_CHECK_MAX_FAILS', '1')),
    )

@app.route('/visualizer')
def visualizer():
    """Live interview + session state visualizer"""
    username = get_current_user().username if current_user.is_authenticated else 'agent'
    return render_template('visualizer.html', username=username)

@app.route('/pilot')
@login_required
def pilot_viewer():
    """Pilot conversation replay viewer"""
    return render_template('pilot.html', username=get_current_user().username)

PILOT_DIR = os.path.join(_root, 'pilot')

@app.route('/api/pilot-data', methods=['GET'])
@login_required
def pilot_data():
    """List pilot sessions or return messages + portrait for a specific session."""
    import re
    user_id = request.args.get('user_id')
    session_idx = request.args.get('session')

    if not user_id or session_idx is None:
        # List available sessions
        sessions = {}
        if os.path.isdir(PILOT_DIR):
            for uid in sorted(os.listdir(PILOT_DIR)):
                logs_dir = os.path.join(PILOT_DIR, uid, 'execution_logs')
                if not os.path.isdir(logs_dir):
                    continue
                sess_names = sorted(
                    d for d in os.listdir(logs_dir)
                    if os.path.isdir(os.path.join(logs_dir, d)) and d.startswith('session_')
                    and os.path.exists(os.path.join(logs_dir, d, 'chat_history.log'))
                )
                if sess_names:
                    sessions[uid] = sess_names
        return jsonify({'success': True, 'sessions': sessions})

    # Return a specific session
    log_path = os.path.join(PILOT_DIR, user_id, 'execution_logs', session_idx, 'chat_history.log')
    agenda_path = os.path.join(PILOT_DIR, user_id, 'execution_logs', session_idx, 'session_agenda.json')
    portrait_path = os.path.join(PILOT_DIR, user_id, 'user_portrait.json')

    if not os.path.exists(log_path):
        return jsonify({'success': False, 'error': 'Session not found'}), 404

    # Parse chat_history.log
    line_re = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ - INFO - (Interviewer|User): (.*)')
    messages = []
    current = None
    with open(log_path, 'r') as f:
        for line in f:
            m = line_re.match(line)
            if m:
                if current:
                    messages.append(current)
                role = 'bot' if m.group(1) == 'Interviewer' else 'user'
                current = {'id': str(len(messages)), 'role': role, 'content': m.group(2).rstrip()}
            elif current:
                current['content'] += '\n' + line.rstrip()
    if current:
        messages.append(current)

    # Load portrait
    portrait = {}
    if os.path.exists(agenda_path):
        try:
            with open(agenda_path) as f:
                agenda = json.load(f)
            portrait = agenda.get('user_portrait', {})
        except Exception:
            pass
    if not portrait and os.path.exists(portrait_path):
        try:
            with open(portrait_path) as f:
                portrait = json.load(f)
        except Exception:
            pass

    return jsonify({'success': True, 'messages': messages, 'user_portrait': portrait})

# =============================================================================
# API ENDPOINTS - PROTECTED (REQUIRE LOGIN)
# All endpoints that handle interview data need @login_required
# =============================================================================

@app.route('/api/start-session', methods=['POST'])
@login_required  # PROTECTED
def start_session():
    """Initialize a new interview session using authenticated user's ID"""
    user_id = get_current_user().id
    
    # DEBUG: Log every request
    call_stack = ''.join(traceback.format_stack()[-3:-1])
    app.logger.info(f"[DEBUG] start-session called by user {get_current_user().username}")
    print(f"[DEBUG] start-session request from {get_current_user().username} (user_id: {user_id})")
    
    # Check if user already has an active in-progress session
    for token, wrapper in active_sessions.items():
        if wrapper.user_id == user_id and wrapper.interview_session.session_in_progress:
            # Return existing session instead of creating duplicate
            app.logger.info(f"Returning existing session {token} for user {get_current_user().username}")
            print(f"[Session] Reusing existing session {token} for {get_current_user().username}")
            existing = wrapper.interview_session
            session_started_at, session_started_at_epoch_ms = _serialize_session_start_time(
                getattr(existing, '_session_start_time', None)
            )
            existing_available_time = _normalize_available_time_minutes(
                getattr(existing, '_available_time', None),
                fallback=DEFAULT_AVAILABLE_TIME_MINUTES,
            )
            if getattr(existing, '_available_time', None) != existing_available_time:
                app.logger.warning(
                    "Normalizing existing session available_time from %r to %r for user %s",
                    getattr(existing, '_available_time', None),
                    existing_available_time,
                    get_current_user().username,
                )
                existing._available_time = existing_available_time
                existing.session_agenda.available_time_minutes = existing_available_time
            return jsonify({
                'success': True,
                'session_token': token,
                'session_id': existing.session_id,
                'user_id': user_id,
                'username': get_current_user().username,
                'message': 'Using existing session',
                'was_existing': True,
                'available_time': existing_available_time,
                'session_started_at': session_started_at,
                'session_started_at_epoch_ms': session_started_at_epoch_ms,
            })
    
    # Create new session only if none exists
    data = request.get_json(silent=True) or {}
    session_type = data.get("session_type", os.getenv("SESSION_TYPE", "intake"))
    raw_available_time = data.get("available_time")  # minutes, from pre-interview question
    available_time = _normalize_available_time_minutes(
        raw_available_time,
        fallback=DEFAULT_AVAILABLE_TIME_MINUTES if raw_available_time is not None else None,
    )
    if raw_available_time is not None and available_time != raw_available_time:
        app.logger.warning(
            "Normalizing requested available_time from %r to %r for user %s",
            raw_available_time,
            available_time,
            get_current_user().username,
        )
    interview_session, session_token = create_interview_session(user_id=user_id, session_type=session_type,
                                                                 available_time=available_time)

    app.logger.info(f"Session created: {session_token} | User: {get_current_user().username} ({user_id})")
    print(f"[Session] Created NEW session {session_token} for user {get_current_user().username}")

    session_started_at, session_started_at_epoch_ms = _serialize_session_start_time(
        getattr(interview_session, '_session_start_time', None)
    )

    return jsonify({
        'success': True,
        'session_token': session_token,
        'session_id': interview_session.session_id,
        'user_id': user_id,
        'username': get_current_user().username,
        'message': 'Session started successfully',
        'was_existing': False,
        'available_time': available_time,
        'session_started_at': session_started_at,
        'session_started_at_epoch_ms': session_started_at_epoch_ms,
    })

@app.route('/api/start-agent-session', methods=['POST'])
def start_agent_session():
    """Start an autonomous session with a simulated user (UserAgent) for testing/visualization."""
    data = request.get_json(silent=True) or {}
    session_type = data.get("session_type", os.getenv("SESSION_TYPE", "intake"))

    viewer = get_current_user().username if current_user.is_authenticated else 'agent'

    # sim_user_id must come from the request body; fall back to logged-in user id only if authenticated
    sim_user_id = data.get("sim_user_id") or (get_current_user().id if current_user.is_authenticated else None)
    if not sim_user_id:
        return jsonify({'success': False, 'error': 'sim_user_id is required'}), 400

    # Check if there's already an active agent session for this sim_user_id
    for token, wrapper in active_sessions.items():
        if wrapper.user_id == sim_user_id:
            app.logger.info(f"Returning existing agent session {token} for sim_user {sim_user_id}")
            return jsonify({
                'success': True,
                'session_token': token,
                'session_id': wrapper.interview_session.session_id,
                'user_id': sim_user_id,
                'username': viewer,
                'message': 'Using existing agent session',
                'was_existing': True,
                'agent_mode': True,
            })

    interview_session, session_token = create_interview_session(
        user_id=sim_user_id,
        session_type=session_type,
        interaction_mode='agent',
    )

    app.logger.info(f"Agent session created: {session_token} | sim_user: {sim_user_id} | watcher: {viewer}")
    print(f"[AgentSession] Created agent session {session_token} simulating user {sim_user_id}")

    return jsonify({
        'success': True,
        'session_token': session_token,
        'session_id': interview_session.session_id,
        'user_id': sim_user_id,
        'username': viewer,
        'message': 'Agent session started successfully',
        'was_existing': False,
        'agent_mode': True,
    })

@app.route('/api/agent-control', methods=['POST'])
@agent_or_login_required
def agent_control():
    """Pause, resume, or step the simulated user agent."""
    data = request.get_json(silent=True) or {}
    session_token = data.get('session_token')
    action = data.get('action')  # 'pause' | 'resume' | 'step'

    session = get_session(session_token)
    if not session:
        return jsonify({'success': False, 'error': 'Invalid or expired session'}), 400

    if action == 'pause':
        session.paused = True
    elif action == 'resume':
        session.paused = False
        session.step_requested = False
    elif action == 'step':
        session.paused = True   # stay paused after this step
        session.step_requested = True
    else:
        return jsonify({'success': False, 'error': f'Unknown action: {action}'}), 400

    return jsonify({'success': True, 'paused': session.paused})

@app.route('/api/send-message', methods=['POST'])
@agent_or_login_required
def send_message():
    """Send a text message to the interview session"""
    data = request.json
    session_token = data.get('session_token')
    user_message = data.get('message')

    session = get_session(session_token)
    if not session:
        return jsonify({
            'success': False,
            'error': 'Invalid or expired session'
        }), 400

    if not session.session_in_progress:
        return jsonify({
            'success': False,
            'error': 'Session has ended',
            'session_completed': True
        }), 400

    turn_id = uuid.uuid4().hex
    api_received_at = datetime.now()
    metadata = {
        "turn_id": turn_id,
        "api_received_at": api_received_at.isoformat(),
        "transport": "text",
    }
    _register_pending_turn(
        session_token,
        turn_id,
        {
            "api_received_at": api_received_at.isoformat(),
            "user_message_length": len(str(user_message or "")),
            "transport": "text",
        },
    )

    wrapper = get_session_wrapper(session_token)
    if wrapper and hasattr(wrapper, 'loop'):
        wrapper.loop.call_soon_threadsafe(
            wrapper.interview_session.user.add_user_message,
            user_message,
            metadata,
        )
    else:
        session.user.add_user_message(user_message, metadata=metadata)

    return jsonify({
        'success': True,
        'message': 'Message queued successfully',
        'turn_id': turn_id,
    })

@app.route('/api/send-voice', methods=['POST'])
@login_required  # PROTECTED
def send_voice():
    """Send a voice message to the interview session"""
    session_token = request.form.get('session_token')
    audio_file = request.files.get('audio')

    if not audio_file:
        return jsonify({
            'success': False,
            'error': 'No audio file provided'
        }), 400

    session = get_session(session_token)
    if not session:
        return jsonify({
            'success': False,
            'error': 'Invalid or expired session'
        }), 400

    orig_filename = audio_file.filename or 'recording.webm'
    ext = Path(orig_filename).suffix or '.webm'
    temp_audio_path = Path(f"temp_audio_{uuid.uuid4().hex}{ext}")
    audio_file.save(temp_audio_path)

    try:
        transcribed_text = transcribe_audio_to_text(temp_audio_path)

        transcribe_only = request.form.get('transcribe_only', 'false').lower() == 'true'
        if not transcribe_only:
            turn_id = uuid.uuid4().hex
            api_received_at = datetime.now()
            metadata = {
                "turn_id": turn_id,
                "api_received_at": api_received_at.isoformat(),
                "transport": "voice",
            }
            _register_pending_turn(
                session_token,
                turn_id,
                {
                    "api_received_at": api_received_at.isoformat(),
                    "user_message_length": len(str(transcribed_text or "")),
                    "transport": "voice",
                },
            )
            wrapper = get_session_wrapper(session_token)
            if wrapper and hasattr(wrapper, 'loop'):
                wrapper.loop.call_soon_threadsafe(
                    wrapper.interview_session.user.add_user_message,
                    transcribed_text,
                    metadata,
                )
            else:
                session.user.add_user_message(transcribed_text, metadata=metadata)

        return jsonify({
            'success': True,
            'transcribed_text': transcribed_text,
            'message': 'Transcription complete' if transcribe_only else 'Voice message processed successfully'
        })
    except Exception as e:
        app.logger.error(f"[send_voice] Transcription error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        if temp_audio_path.exists():
            temp_audio_path.unlink()

def _prestart_tts(session_token: str, message_id: str, text: str, wrapper) -> None:
    """Kick off TTS generation in background immediately when a message is delivered."""
    cache = session_audio_cache.setdefault(session_token, {})
    if message_id in cache:
        return  # Already started or done
    cache[message_id] = {'status': 'pending', 'data': None, 'error': None, 'timestamp': time.time()}

    async def _synth():
        try:
            def _blocking():
                out = Path(f"temp_speech_{uuid.uuid4().hex}.mp3")
                try:
                    tts_engine.text_to_speech(text=text, output_path=str(out))
                    return out.read_bytes()
                finally:
                    if out.exists():
                        out.unlink(missing_ok=True)
            data = await wrapper.loop.run_in_executor(None, _blocking)
            cache[message_id] = {'status': 'ready', 'data': data, 'error': None, 'timestamp': time.time()}
            app.logger.info(f"TTS pre-generated {len(data)} bytes for message {message_id}")
        except Exception as e:
            cache[message_id] = {'status': 'failed', 'data': None, 'error': str(e), 'timestamp': time.time()}
            app.logger.error(f"TTS pre-generation failed for {message_id}: {e}")

    asyncio.run_coroutine_threadsafe(_synth(), wrapper.loop)


@app.route('/api/get-messages', methods=['GET'])
@agent_or_login_required
def get_messages():
    """Get new messages from the session (polling endpoint)"""
    session_token = request.args.get('session_token')

    session = get_session(session_token)
    if not session:
        print(f"[get_messages] Invalid session_token={session_token}")
        return jsonify({
            'success': False,
            'error': 'Invalid or expired session',
            'active_sessions_count': len(active_sessions)
        }), 400

    messages = []
    full_history = request.args.get('full', 'false').lower() == 'true'
    has_user_buffer = session.user and (
        hasattr(session.user, 'get_new_messages') or
        hasattr(session.user, '_message_buffer') or
        hasattr(session.user, 'get_and_clear_messages')
    )
    _deliverable_types = {
        'conversation',
        'time_split_widget',
        'ai_usage_widget',
        'ai_task_widget',
        'feedback_widget',
        'profile_confirm_widget',
        'task_validation_widget',
    }
    if has_user_buffer and full_history:
        # Reconnect: replay deliverable messages from full chat_history so the
        # client can restore its current state.
        # Keep feedback/profile/AI-usage widget triggers; skip time-split on reconnect.
        messages = [
            {
                'id': m.id,
                'role': m.role,
                'content': m.content,
                'type': m.type,
                'timestamp': m.timestamp.isoformat(),
                'metadata': getattr(m, "metadata", {}) if isinstance(getattr(m, "metadata", {}), dict) else {},
            }
            for m in session.chat_history
            if m.type in _deliverable_types and m.type != 'time_split_widget'
        ]
    elif has_user_buffer:
        if hasattr(session.user, 'get_new_messages'):
            messages = session.user.get_new_messages() or []
        elif hasattr(session.user, 'get_and_clear_messages'):
            messages = session.user.get_and_clear_messages() or []
        elif hasattr(session.user, '_message_buffer'):
            lock = getattr(session.user, '_lock', None)
            if lock:
                with lock:
                    messages = list(getattr(session.user, '_message_buffer', []))
                    session.user._message_buffer.clear()
            else:
                messages = list(getattr(session.user, '_message_buffer', []))
                session.user._message_buffer.clear()
    else:
        # Agent mode: UserAgent has no buffer — serve new messages directly from chat_history
        if full_history:
            chat_history_offsets[session_token] = 0
        offset = chat_history_offsets.get(session_token, 0)
        new_msgs = session.chat_history[offset:]
        chat_history_offsets[session_token] = offset + len(new_msgs)
        messages = [
            {
                'id': m.id,
                'role': m.role,
                'content': m.content,
                'type': m.type,
                'timestamp': m.timestamp.isoformat(),
                'metadata': getattr(m, "metadata", {}) if isinstance(getattr(m, "metadata", {}), dict) else {},
            }
            for m in new_msgs
            if m.type in _deliverable_types
        ]

    is_session_done = session.session_completed

    if not is_session_done:
        current_turns = getattr(session, 'turns', 0)
        max_turns = getattr(session, 'max_turns', float('inf'))
        if current_turns is not None and max_turns is not None and current_turns >= max_turns:
            is_session_done = True
        elif not session.session_in_progress and len(session.chat_history) > 0:
            is_session_done = True

    # Session completion is signaled via data.session_completed in the JSON response;
    # the frontend renders its own end-of-session banner.

    # Log per-turn delivery latency once an interviewer conversation message is
    # actually delivered to a polling client.
    if messages:
        delivered_at = datetime.now()
        history_by_id = {
            getattr(m, "id", None): m for m in session.chat_history if getattr(m, "id", None)
        }
        delivered_keys = delivered_turn_messages_by_session.setdefault(session_token, set())
        pending_turns = pending_turns_by_session.setdefault(session_token, OrderedDict())
        session_user_id = getattr(session, "user_id", None)
        session_id = getattr(session, "session_id", None)
        eval_logger = None
        if session_user_id is not None and session_id is not None:
            eval_logger = EvaluationLogger(user_id=session_user_id, session_id=session_id)

        for msg in messages:
            if eval_logger is None:
                break
            if msg.get("role") not in {"Interviewer", "assistant"}:
                continue
            if str(msg.get("type")) != "conversation":
                continue

            message_id = str(msg.get("id") or "").strip()
            if not message_id:
                continue
            source_msg = history_by_id.get(message_id)
            if source_msg is None:
                continue

            source_meta = getattr(source_msg, "metadata", {})
            meta = source_meta if isinstance(source_meta, dict) else {}
            turn_id = str(meta.get("turn_id", "")).strip()
            if not turn_id:
                continue

            delivered_key = f"{turn_id}:{message_id}"
            if delivered_key in delivered_keys:
                continue
            delivered_keys.add(delivered_key)

            user_message_id = str(meta.get("paired_user_message_id", "")).strip()
            user_chat_history_at = _parse_iso_datetime(meta.get("paired_user_timestamp"))
            if user_chat_history_at is None and user_message_id:
                user_msg = history_by_id.get(user_message_id)
                if user_msg is not None:
                    user_chat_history_at = user_msg.timestamp
            if user_chat_history_at is None:
                continue

            api_received_at = _parse_iso_datetime(meta.get("api_received_at"))
            if api_received_at is None:
                pending = pending_turns.get(turn_id, {})
                api_received_at = _parse_iso_datetime(pending.get("api_received_at"))

            user_len = int(meta.get("user_message_length", 0) or 0)
            if user_len <= 0:
                pending = pending_turns.get(turn_id, {})
                user_len = int(pending.get("user_message_length", 0) or 0)

            transport = str(meta.get("transport", "")).strip() or "text"
            if transport == "text":
                pending = pending_turns.get(turn_id, {})
                transport = str(pending.get("transport", "text"))

            eval_logger.log_turn_latency_breakdown(
                turn_id=turn_id,
                user_message_id=user_message_id,
                assistant_message_id=message_id,
                user_api_received_at=api_received_at,
                user_chat_history_at=user_chat_history_at,
                assistant_generated_at=source_msg.timestamp,
                assistant_delivered_at=delivered_at,
                user_message_length=user_len,
                assistant_message_length=len(str(source_msg.content or "")),
                transport=transport,
            )
            pending_turns.pop(turn_id, None)

    # Pre-start TTS generation for interviewer messages as soon as they are delivered,
    # so audio is ready (or nearly ready) by the time the client explicitly requests it.
    wrapper = get_session_wrapper(session_token)
    if wrapper and hasattr(wrapper, 'loop'):
        for msg in messages:
            if msg.get('role') in ('Interviewer', 'assistant') and msg.get('id') and msg.get('content'):
                _prestart_tts(session_token, msg['id'], msg['content'], wrapper)

    end_reason = getattr(getattr(session, 'session_agenda', None), 'end_reason', 'completed')
    job_description_ready = bool(_extract_job_description(session))

    return jsonify({
        'success': True,
        'messages': messages,
        'session_active': session.session_in_progress,
        'session_completed': is_session_done,
        'end_reason': end_reason,
        'job_description_ready': job_description_ready,
    })

@app.route('/api/acknowledge-messages', methods=['POST'])
@agent_or_login_required
def acknowledge_messages():
    """Mark messages as acknowledged by the client"""
    data = request.json
    session_token = data.get('session_token')
    message_ids = data.get('message_ids', [])

    session = get_session(session_token)
    if not session:
        return jsonify({
            'success': False,
            'error': 'Invalid or expired session'
        }), 400

    if session.user and hasattr(session.user, '_message_buffer'):
        lock = getattr(session.user, '_lock', None)
        if lock:
            with lock:
                buffer = getattr(session.user, '_message_buffer', [])
                session.user._message_buffer = [
                    m for m in buffer 
                    if m.get('id') not in message_ids
                ]
        
    return jsonify({'success': True})

@app.route('/api/get-voice-response', methods=['GET'])
@login_required  # PROTECTED
def get_voice_response():
    """Get the latest interviewer message as voice audio"""
    session_token = request.args.get('session_token')
    message_id = request.args.get('message_id')

    session = get_session(session_token)
    if not session:
        return jsonify({
            'success': False,
            'error': 'Invalid or expired session'
        }), 400

    if not message_id:
        return jsonify({
            'success': False,
            'error': 'message_id required'
        }), 400

    target_msg = None
    for m in session.chat_history:
        if hasattr(m, 'id') and m.id == message_id:
            target_msg = m
            break
    
    if not target_msg:
        return jsonify({
            'success': False,
            'error': 'Message not found'
        }), 404

    cache = session_audio_cache.setdefault(session_token, {})
    entry = cache.get(message_id)

    # Check cache entry status
    if isinstance(entry, dict):
        # New format: {'status': 'pending'|'ready'|'failed', 'data': bytes|None, 'error': str|None}
        status = entry.get('status')
        if status == 'ready' and entry.get('data'):
            return Response(entry['data'], mimetype='audio/mpeg')
        elif status == 'failed':
            return jsonify({
                'success': False,
                'error': f"TTS generation failed: {entry.get('error', 'Unknown error')}"
            }), 500
        elif status == 'pending':
            return ('', 202)  # Still generating
    elif isinstance(entry, (bytes, bytearray)):
        # Legacy format support
        return Response(entry, mimetype='audio/mpeg')
    elif entry == 'pending':
        # Legacy format support
        return ('', 202)

    # No cache entry exists — start generation (also triggered from get_messages,
    # but guard here in case the client requests TTS before the next poll cycle).
    wrapper = get_session_wrapper(session_token)

    if not wrapper or not hasattr(wrapper, 'loop'):
        cache[message_id] = {
            'status': 'failed',
            'data': None,
            'error': 'Session loop not available'
        }
        return jsonify({
            'success': False,
            'error': 'Session not properly initialized'
        }), 500

    _prestart_tts(session_token, message_id, target_msg.content, wrapper)

    return ('', 202)  # Tell client to poll again

@app.route('/api/stream-voice-response', methods=['GET'])
@login_required
def stream_voice_response():
    """Stream TTS audio for a message, starting playback before generation is complete."""
    from src.utils.speech.text_to_speech import CartesiaTTS
    session_token = request.args.get('session_token')
    message_id = request.args.get('message_id')

    session = get_session(session_token)
    if not session or not message_id:
        return jsonify({'success': False, 'error': 'Invalid request'}), 400

    cache = session_audio_cache.setdefault(session_token, {})
    entry = cache.get(message_id)

    # Serve from cache if already ready (avoids hitting Cartesia twice on replay)
    if isinstance(entry, dict) and entry.get('status') == 'ready' and entry.get('data'):
        mimetype = entry.get('mimetype', 'audio/mpeg')
        return Response(entry['data'], mimetype=mimetype)

    # Find message text
    target_msg = next(
        (m for m in session.chat_history if hasattr(m, 'id') and m.id == message_id),
        None
    )
    if not target_msg:
        return jsonify({'success': False, 'error': 'Message not found'}), 404

    if not isinstance(tts_engine, CartesiaTTS):
        # Non-Cartesia provider: kick off async generation, then stream bytes
        # once ready so the browser's <audio> element receives real audio data.
        wrapper = get_session_wrapper(session_token)
        if wrapper and hasattr(wrapper, 'loop'):
            _prestart_tts(session_token, message_id, target_msg.content, wrapper)

        def wait_and_serve():
            deadline = time.time() + 30
            while time.time() < deadline:
                entry = cache.get(message_id)
                if isinstance(entry, dict):
                    if entry.get('status') == 'ready' and entry.get('data'):
                        yield entry['data']
                        return
                    elif entry.get('status') == 'failed':
                        return
                time.sleep(0.2)

        return Response(stream_with_context(wait_and_serve()), mimetype='audio/mpeg')

    # Mark as streaming so _prestart_tts skips this message
    cache[message_id] = {'status': 'streaming', 'data': None, 'error': None, 'timestamp': time.time()}

    def generate():
        chunks = []
        try:
            for chunk in tts_engine.stream_tts_chunks(target_msg.content):
                chunks.append(chunk)
                yield chunk
            full_audio = b''.join(chunks)
            cache[message_id] = {
                'status': 'ready', 'data': full_audio,
                'mimetype': 'audio/mpeg', 'error': None, 'timestamp': time.time()
            }
            app.logger.info(f"Streamed and cached {len(full_audio)} bytes for message {message_id}")
        except Exception as e:
            cache[message_id] = {'status': 'failed', 'data': None, 'error': str(e), 'timestamp': time.time()}
            app.logger.error(f"Streaming TTS failed for {message_id}: {e}")

    return Response(stream_with_context(generate()), mimetype='audio/mpeg')

@app.route('/api/end-session', methods=['POST'])
@agent_or_login_required
def end_session():
    """Begin ending the session by emitting the feedback widget. The session
    truly ends once /api/submit-feedback is called."""
    data = request.json
    session_token = data.get('session_token')

    wrapper = get_session_wrapper(session_token)
    if not wrapper:
        return jsonify({
            'success': False,
            'error': 'Invalid or expired session'
        }), 400

    session = wrapper.interview_session

    # Emit feedback widget on the session's event loop (widget sending
    # touches asyncio state inside add_message_to_chat_history).
    if hasattr(wrapper, 'loop'):
        wrapper.loop.call_soon_threadsafe(session.trigger_feedback_widget)
    else:
        session.trigger_feedback_widget()

    app.logger.info(f"Session {session_token}: feedback widget emitted by user end-session request")

    return jsonify({
        'success': True,
        'message': 'Feedback widget emitted; session will end after feedback is submitted',
        'session_id': session.session_id,
        'user_id': session.user_id,
        'awaiting_feedback': True,
    })


@app.route('/api/submit-feedback', methods=['POST'])
@agent_or_login_required
def submit_feedback():
    """Persist end-of-session feedback and finalize session completion."""
    data = request.get_json(silent=True) or {}
    session_token = data.get('session_token')
    feedback = data.get('feedback')

    if not session_token or not isinstance(feedback, dict):
        return jsonify({'success': False, 'error': 'session_token and feedback required'}), 400

    wrapper = get_session_wrapper(session_token)
    if not wrapper:
        return jsonify({'success': False, 'error': 'Invalid or expired session'}), 400

    session = wrapper.interview_session

    if hasattr(wrapper, 'loop'):
        wrapper.loop.call_soon_threadsafe(session.submit_feedback, feedback)
    else:
        session.submit_feedback(feedback)

    app.logger.info(f"Session {session_token}: feedback submitted, finalizing session")

    return jsonify({
        'success': True,
        'message': 'Feedback received; session finalizing',
        'session_id': session.session_id,
        'user_id': session.user_id,
    })

@app.route('/api/session-status', methods=['GET'])
@login_required  # PROTECTED
def session_status():
    """Get current session status including background task progress"""
    session_token = request.args.get('session_token')

    wrapper = get_session_wrapper(session_token)
    if not wrapper:
        return jsonify({
            'success': False,
            'error': 'Invalid or expired session'
        }), 400

    session = wrapper.interview_session

    # Get background task count if available
    background_tasks_count = 0
    if hasattr(session, '_background_tasks'):
        try:
            import asyncio
            # Try to get count safely
            if hasattr(session, '_background_tasks_lock'):
                # Can't acquire lock in sync context, just get len
                background_tasks_count = len(session._background_tasks)
        except:
            background_tasks_count = 0

    return jsonify({
        'success': True,
        'session_active': session.session_in_progress,
        'session_completed': session.session_completed,
        'background_tasks_remaining': background_tasks_count,
        'message_count': len(session.chat_history),
        'session_id': session.session_id,
        'user_id': session.user_id
    })

@app.route('/api/session-state', methods=['GET'])
@agent_or_login_required
def session_state():
    """Return serialized live session state for the visualizer"""
    session_token = request.args.get('session_token')
    wrapper = get_session_wrapper(session_token)
    if not wrapper:
        return jsonify({'success': False, 'error': 'Invalid or expired session'}), 400

    iv = wrapper.interview_session
    agenda = getattr(iv, 'session_agenda', None)
    memory_bank = getattr(iv, 'memory_bank', None)

    # --- Topics & subtopics ---
    topics = []
    if agenda and agenda.interview_topic_manager:
        for topic in agenda.interview_topic_manager:
            subtopics = []
            for st in topic.required_subtopics.values():
                subtopics.append({
                    'id': st.subtopic_id,
                    'description': st.description,
                    'is_covered': st.is_covered,
                    'notes': list(st.notes),
                    'notes_count': len(st.notes),
                    'questions_count': len(st.questions),
                    'emergent_insights_count': len(st.emergent_insights),
                    'final_summary': st.final_summary,
                    'is_emergent': False,
                    'coverage_criteria': list(st.coverage_criteria),
                    'criteria_coverage': list(st.criteria_coverage),
                })
            for st in topic.emergent_subtopics.values():
                subtopics.append({
                    'id': st.subtopic_id,
                    'description': st.description,
                    'is_covered': st.is_covered,
                    'notes': list(st.notes),
                    'notes_count': len(st.notes),
                    'questions_count': len(st.questions),
                    'emergent_insights_count': len(st.emergent_insights),
                    'final_summary': st.final_summary,
                    'is_emergent': True,
                    'coverage_criteria': list(st.coverage_criteria),
                    'criteria_coverage': list(st.criteria_coverage),
                })
            topics.append({
                'id': topic.topic_id,
                'description': topic.description,
                'subtopics': subtopics,
            })

    # --- Recent memories (last 20) ---
    memories = []
    memory_count = 0
    if memory_bank:
        all_mems = memory_bank.memories
        memory_count = len(all_mems)
        for mem in all_mems[-20:]:
            memories.append({
                'id': mem.id,
                'title': mem.title,
                'text': mem.text[:250],
                'timestamp': mem.timestamp.isoformat() if hasattr(mem.timestamp, 'isoformat') else str(mem.timestamp),
                'subtopic_links': mem.subtopic_links,
                'transcript_references': [
                    {
                        'interview_response': ref.interview_response[:200],
                        'timestamp': ref.timestamp.isoformat() if ref.timestamp else None,
                    }
                    for ref in (mem.transcript_references or [])
                ],
            })

    # --- Chat history as lightweight turn list ---
    turns = []
    for msg in iv.chat_history:
        turns.append({
            'id': getattr(msg, 'id', ''),
            'role': getattr(msg, 'role', ''),
            'content': getattr(msg, 'content', ''),
            'timestamp': getattr(msg, 'timestamp', None).isoformat() if getattr(msg, 'timestamp', None) else '',
        })

    # --- Strategic priorities ---
    strategic_priorities = agenda.strategic_priorities or {} if agenda else {}

    # --- Rollout predictions (simulated exchanges) ---
    rollout_predictions = []
    strategic_planner = getattr(iv, 'strategic_planner', None)
    if strategic_planner:
        strategic_state = getattr(strategic_planner, 'strategic_state', None)
        if strategic_state:
            for rollout in (strategic_state.rollout_predictions or []):
                rollout_predictions.append({
                    'rollout_id': rollout.rollout_id,
                    'utility_score': round(rollout.utility_score, 3),
                    'expected_coverage_delta': round(rollout.expected_coverage_delta, 3),
                    'emergence_potential': round(rollout.emergence_potential, 3),
                    'cost_estimate': rollout.cost_estimate,
                    'predicted_turns': [
                        {
                            'turn_number': t.get('turn_number', '?'),
                            'question': str(t.get('question', '')),
                            'predicted_response': str(t.get('predicted_response', '')),
                            'subtopics_covered': t.get('subtopics_covered', []),
                            'emergence_potential': t.get('emergence_potential', 0.0),
                            'strategic_rationale': str(t.get('strategic_rationale', '')),
                        }
                        for t in rollout.predicted_turns
                    ],
                })

    # --- Emergent insights ---
    emergent_insights = []
    if agenda:
        for ins in (agenda.emergent_insights or []):
            if isinstance(ins, dict):
                emergent_insights.append(ins)
            else:
                emergent_insights.append(ins.to_dict() if hasattr(ins, 'to_dict') else str(ins))

    return jsonify({
        'success': True,
        'topics': topics,
        'memories': memories,
        'memory_count': memory_count,
        'turns': turns,
        'turn_count': len(iv.chat_history),
        'user_portrait': agenda.user_portrait if agenda else {},
        'last_week_snapshot': agenda.last_week_snapshot if agenda else {},
        'strategic_priorities': strategic_priorities,
        'rollout_predictions': rollout_predictions,
        'emergent_insights': emergent_insights,
        'session_type': getattr(iv, 'session_type', 'intake'),
        'session_completed': getattr(iv, 'session_completed', False),
        'session_in_progress': getattr(iv, 'session_in_progress', False),
        'timestamp': __import__('datetime').datetime.now().isoformat(),
    })





@app.route('/api/time-split-ready', methods=['POST'])
@agent_or_login_required
def time_split_ready():
    """Return a freshness-gated payload for the time-split widget.

    Waits for scribe processing to settle, ensures portrait freshness through
    the latest memory count, then returns both portrait and organized task tree.
    """
    data = request.get_json(silent=True) or {}
    session_token = data.get('session_token')
    if not session_token:
        return jsonify({'success': False, 'error': 'session_token required'}), 400

    wrapper = get_session_wrapper(session_token)
    if not wrapper:
        return jsonify({'success': False, 'error': 'Invalid or expired session'}), 400

    try:
        timeout_seconds = float(data.get('timeout_seconds', 35))
    except (TypeError, ValueError):
        timeout_seconds = 35.0
    timeout_seconds = max(1.0, min(timeout_seconds, 120.0))

    iv = wrapper.interview_session

    async def _build_payload():
        loop = asyncio.get_running_loop()
        scribe = getattr(iv, 'session_scribe', None)

        wait_start = loop.time()
        while scribe is not None and getattr(scribe, 'processing_in_progress', False):
            if loop.time() - wait_start >= timeout_seconds:
                return {'ready': False}
            await asyncio.sleep(0.1)
        scribe_wait_s = loop.time() - wait_start

        target_memory_count = len(getattr(iv.memory_bank, 'memories', []))
        portrait_start = loop.time()
        refreshed = await iv.ensure_user_portrait_fresh(
            min_memory_count=target_memory_count,
            wait_for_inflight=True,
        )
        portrait_wait_s = loop.time() - portrait_start

        agenda = getattr(iv, 'session_agenda', None)
        portrait = agenda.user_portrait if agenda and isinstance(agenda.user_portrait, dict) else {}
        raw_tasks = portrait.get('Task Inventory') if isinstance(portrait, dict) else []
        tasks = [str(t).strip() for t in (raw_tasks or []) if str(t).strip()]

        app.logger.info(
            "[time_split_ready] scribe_wait=%.2fs portrait_wait=%.2fs refreshed=%s tasks=%d",
            scribe_wait_s,
            portrait_wait_s,
            refreshed,
            len(tasks),
        )
        return {
            'ready': True,
            'user_portrait': portrait,
            'task_count': len(tasks),
        }

    future = asyncio.run_coroutine_threadsafe(_build_payload(), wrapper.loop)
    try:
        payload = future.result(timeout=timeout_seconds + 10.0)
    except concurrent.futures.TimeoutError:
        future.cancel()
        return jsonify({
            'success': False,
            'error': 'Timed out waiting for fresh portrait',
        }), 504
    except Exception as e:
        app.logger.error(f"[time_split_ready] Failed: {e}", exc_info=True)
        return jsonify({'success': False, 'error': 'Failed to build time-split data'}), 500

    if not payload.get('ready'):
        return jsonify({
            'success': False,
            'error': 'Timed out waiting for fresh portrait',
        }), 504

    return jsonify({
        'success': True,
        'user_portrait': payload.get('user_portrait', {}),
        'task_count': int(payload.get('task_count', 0)),
    })


@app.route('/api/update-portrait', methods=['POST'])
@agent_or_login_required
def update_portrait():
    data = request.get_json()
    session_token = data.get('session_token')
    portrait = data.get('portrait')
    if not session_token or portrait is None:
        return jsonify({'success': False, 'error': 'session_token and portrait required'}), 400
    wrapper = get_session_wrapper(session_token)
    if not wrapper:
        return jsonify({'success': False, 'error': 'Invalid or expired session'}), 400
    iv = wrapper.interview_session
    agenda = getattr(iv, 'session_agenda', None)
    if agenda:
        portrait = normalize_user_portrait(portrait)
        agenda.user_portrait = portrait
        portrait_path = os.path.join(os.getenv('LOGS_DIR', 'logs'), iv.user_id, 'user_portrait.json')
        os.makedirs(os.path.dirname(portrait_path), exist_ok=True)
        with open(portrait_path, 'w') as f:
            json.dump(portrait, f, indent=2)
    return jsonify({'success': True})

# Pick best available model based on credentials present
def _default_task_gen_model() -> str:
    if os.getenv("ANTHROPIC_API_KEY"):
        return "claude-sonnet-4-6"       # direct Anthropic API
    if os.getenv("GCP_PROJECT") and os.getenv("GCP_REGION"):
        return "claude-3-7-sonnet"       # Vertex AI
    return "gpt-4o"                      # OpenAI fallback

_TASK_GEN_MODEL = os.getenv("TASK_GEN_MODEL", _default_task_gen_model())
_TASK_BATCH_SIZE = 3    # tasks shown per batch; each LLM call generates exactly this many
_TASK_MIN_BATCHES = 10  # LLM cannot stop before this many batches (~30 tasks min)
_TASK_MAX_BATCHES = 30  # hard cap to avoid infinite generation


def _parse_tasks_json(raw: str) -> dict:
    """Extract and parse the last valid JSON object containing a 'tasks' key.

    Handles LLM responses that include self-correction text or extra content
    by scanning from right to left for the most complete valid JSON object.
    """
    import re as _re
    import json as _json
    # Strip markdown fences
    raw = _re.sub(r"^```[a-z]*\n?", "", raw.strip())
    raw = _re.sub(r"\n?```$", "", raw).strip()
    # Try each { position from right to left using raw_decode
    for m in reversed(list(_re.finditer(r'\{', raw))):
        try:
            obj, _ = _json.JSONDecoder().raw_decode(raw, m.start())
            if isinstance(obj, dict) and 'tasks' in obj:
                return obj
        except Exception:
            continue
    # Last resort: plain json.loads on the full string
    return _json.loads(raw)


def _extract_job_description(iv) -> str:
    """Return the user's first conversational answer from chat history."""
    for msg in iv.chat_history:
        if msg.role == "User" and msg.type == "conversation" and msg.content.strip():
            return msg.content.strip()
    return ""


def _extract_user_mentioned_tasks(iv) -> list[str]:
    """Extract tasks explicitly mentioned in the conversation via the user portrait.

    Falls back to the raw chat transcript if the portrait has no Task Inventory yet.
    """
    # Prefer portrait Task Inventory (already parsed by scribe)
    try:
        portrait = iv.session_agenda.user_portrait
        if isinstance(portrait, dict):
            tasks = portrait.get("Task Inventory") or []
            result = [str(t).strip() for t in tasks if str(t).strip()]
            if result:
                return result
    except Exception:
        pass

    # Fallback: pull all user messages from chat history
    msgs = [
        msg.content.strip()
        for msg in iv.chat_history
        if msg.role == "User" and getattr(msg, "type", "conversation") == "conversation"
        and msg.content.strip()
    ]
    return msgs



_SESSION_PERSPECTIVES = [
    "Emphasize tasks that are recurring and form the backbone of the daily or weekly routine.",
    "Emphasize tasks that involve coordinating with, communicating with, or depending on other people.",
    "Emphasize tasks that are less frequent but carry high stakes or require significant effort.",
    "Emphasize tasks that involve documentation, tracking, or managing information.",
    "Emphasize tasks that involve external parties such as clients, partners, vendors, or the public.",
    "Emphasize tasks that are often invisible or taken for granted but are essential to the role.",
    "Emphasize tasks that require specialized judgment, expertise, or domain knowledge.",
    "Emphasize tasks related to planning, prioritizing, and managing workloads or timelines.",
    "Emphasize tasks that involve using specific tools, systems, or technology.",
    "Emphasize tasks that have direct impact on outcomes, quality, or the work of others.",
    "Emphasize tasks that involve learning, staying current, or improving skills and knowledge.",
    "Emphasize tasks that come up reactively — in response to problems, requests, or unexpected events.",
]


def _session_perspective(session_token: str) -> str:
    """Pick a stable perspective for this session based on the session token."""
    import hashlib
    h = int(hashlib.md5(session_token.encode()).hexdigest(), 16)
    return _SESSION_PERSPECTIVES[h % len(_SESSION_PERSPECTIVES)]


def _llm_generate_task_batch(
    job_description: str,
    prior_tasks: list[str],
    batch_index: int,
    session_perspective: str = "",
    mentioned_tasks: list[str] | None = None,
) -> tuple[list[str], bool]:
    """Generate the next batch of tasks, ordered by importance, deduped against prior_tasks."""
    import re as _re
    from src.utils.llm.engines import get_engine, invoke_engine

    prior_block = ""
    if prior_tasks:
        prior_list = "\n".join(f"- {t}" for t in prior_tasks)
        prior_block = (
            f"\nTasks already shown (STRICT: do NOT repeat or rephrase any of these — "
            f"treat semantically similar tasks as duplicates):\n{prior_list}\n"
        )

    # On batch 0, surface tasks explicitly mentioned in the conversation first.
    mentioned_block = ""
    if batch_index == 0 and mentioned_tasks:
        ml = "\n".join(f"- {t}" for t in mentioned_tasks[:20])
        mentioned_block = (
            f"\nTasks this person explicitly mentioned doing:\n{ml}\n"
            f"IMPORTANT: Your output MUST include all of these (verbatim or lightly cleaned up "
            f"to fit the 5–12 word action+object format), unless they are already in the "
            f"'Tasks already shown' list above. Do not omit any.\n"
        )

    stop_hint = (
        "Set has_more to false only when all meaningful, distinct task areas are covered "
        "and there are genuinely no more important tasks to add. Otherwise set has_more to true."
    )

    _CORE_BATCHES = 2  # first 2 batches (6 tasks) = core, rest = diversify
    if batch_index < _CORE_BATCHES:
        rank = batch_index + 1
        coverage_hint = (
            f"Return the {_TASK_BATCH_SIZE} {'most' if batch_index == 0 else 'next most'} important "
            f"core tasks for this role — the work this person spends the most time on. "
            f"Rank strictly by centrality and frequency (batch {rank} of {_CORE_BATCHES} core batches)."
        )
    else:
        coverage_hint = (
            f"The first {_CORE_BATCHES * _TASK_BATCH_SIZE} tasks covered the core of this role. "
            f"Now return {_TASK_BATCH_SIZE} tasks from a DIFFERENT area not yet represented — "
            "vary the type of work: coordination, documentation, external-facing, reactive, specialized, etc. "
            "Think about what a complete picture of this occupation looks like across all dimensions."
        )

    perspective_line = f"Session focus: {session_perspective}\n" if session_perspective else ""

    prompt = (
        f"Occupation context: \"{job_description}\"\n"
        f"{perspective_line}"
        f"{prior_block}"
        f"{mentioned_block}\n"
        f"{coverage_hint}\n\n"
        "Rules:\n"
        "- Focus on tasks typical of the general occupation/role category, not details specific to this individual\n"
        "- Tasks must be DISTINCT — no two tasks should describe the same activity even with different wording\n"
        "- Each task: 5–12 words, starts with an action verb, sentence case (only first word and proper nouns capitalized)\n"
        "- Be specific — include object and context where helpful\n"
        "  Good: 'Review pull requests and leave code comments'\n"
        "  Bad: 'Review code'\n"
        f"- {stop_hint}\n"
        "- Each task needs a name (5–12 words) and a description (1 sentence, what it involves).\n"
        "- Return ONLY valid JSON (no markdown fences):\n"
        f'  {{"tasks": [{{"name": "Task name here", "description": "One sentence describing what this involves."}}, ...], "has_more": true}}'
    )

    engine = get_engine(model_name=_TASK_GEN_MODEL, temperature=0.9)
    response = invoke_engine(engine, prompt)
    raw = (response.content if hasattr(response, "content") else str(response)).strip()

    try:
        parsed = _parse_tasks_json(raw)
        tasks_raw = parsed.get("tasks") or []
        tasks = []
        for t in tasks_raw:
            if isinstance(t, dict):
                name = str(t.get("name", "")).strip()
                desc = str(t.get("description", "")).strip()
                if name and len(name.split()) >= 3:
                    tasks.append({"name": name, "description": desc})
            elif isinstance(t, str) and len(t.strip().split()) >= 3:
                tasks.append({"name": t.strip(), "description": ""})
        has_more = bool(parsed.get("has_more", True))
    except Exception:
        _JSON_KEYS = {"tasks", "name", "description", "has_more", "ai_type",
                      "capability", "governance", "true", "false", "null"}
        names = _re.findall(r'"([^"]{4,80})"', raw)
        tasks = [
            {"name": n.strip(), "description": ""}
            for n in names
            if n.strip() and n.strip().lower() not in _JSON_KEYS
            and len(n.strip().split()) >= 3
        ]
        has_more = True

    # Fill any missing descriptions in one extra LLM call rather than making
    # the frontend fire per-card requests later.
    missing = [t for t in tasks if not t["description"]]
    if missing:
        numbered = "\n".join(f"{i+1}. {t['name']}" for i, t in enumerate(missing))
        fill_prompt = (
            f'Job context: "{job_description}"\n\n'
            "For each task below write exactly one sentence (under 20 words) describing what it involves. "
            "Start each with a verb. Return ONLY a JSON object mapping task name to description:\n"
            '{"task name": "description", ...}\n\n'
            f"Tasks:\n{numbered}"
        )
        try:
            fill_engine = get_engine(model_name=_TASK_GEN_MODEL, temperature=0.7)
            fill_response = invoke_engine(fill_engine, fill_prompt)
            fill_raw = (fill_response.content if hasattr(fill_response, "content") else str(fill_response)).strip()
            fill_raw = _re.sub(r'^```[a-z]*\n?', '', fill_raw)
            fill_raw = _re.sub(r'\n?```$', '', fill_raw).strip()
            fill_map = json.loads(fill_raw)
            for t in missing:
                t["description"] = str(fill_map.get(t["name"], "")).strip()
        except Exception:
            pass  # leave descriptions empty; frontend batch-fetch is the safety net

    return tasks, has_more


@app.route('/api/generate-tasks', methods=['POST'])
@agent_or_login_required
def generate_tasks():
    """Generate one batch of tasks for the task-validation widget.

    Accepts batch_index (0-based) and prior_tasks (already shown) so each call
    is small (~3 tasks) and fast. The frontend pre-fetches the next batch while
    the user reviews the current one, giving near-zero perceived latency.
    """
    data = request.get_json(silent=True) or {}
    session_token = data.get('session_token')
    batch_index = int(data.get('batch_index', 0))
    prior_tasks = data.get('prior_tasks') or []
    if not isinstance(prior_tasks, list):
        prior_tasks = []

    wrapper = get_session_wrapper(session_token)
    if not wrapper:
        return jsonify({'success': False, 'error': 'Invalid or expired session'}), 400

    iv = wrapper.interview_session
    job_description = _extract_job_description(iv)
    if not job_description:
        return jsonify({'success': False, 'error': 'No job description found in session'}), 400

    # Hard cap: if we've already generated enough batches, stop
    if batch_index >= _TASK_MAX_BATCHES:
        return jsonify({'success': True, 'tasks': [], 'has_more': False})

    try:
        perspective = _session_perspective(session_token or "")
        mentioned = _extract_user_mentioned_tasks(iv) if batch_index == 0 else None
        tasks, has_more = _llm_generate_task_batch(
            job_description, prior_tasks, batch_index, perspective, mentioned
        )
    except Exception as e:
        app.logger.error(f"[generate_tasks] LLM error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

    # Don't let the LLM stop before _TASK_MIN_BATCHES batches
    if batch_index + 1 < _TASK_MIN_BATCHES:
        has_more = True

    # Enforce hard cap
    if batch_index + 1 >= _TASK_MAX_BATCHES:
        has_more = False

    return jsonify({'success': True, 'tasks': tasks, 'has_more': has_more})


@app.route('/api/ai-era-tasks', methods=['POST'])
@agent_or_login_required
def ai_era_tasks():
    """Generate a batch of AI-era tasks. Supports pagination via batch_index + prior_tasks."""
    from src.utils.llm.engines import get_engine, invoke_engine

    _AI_BATCH_SIZE = 3
    _AI_MAX_BATCHES = 6   # up to 18 tasks total

    data = request.get_json(silent=True) or {}
    session_token = data.get('session_token')
    batch_index = int(data.get('batch_index', 0))
    prior_tasks = data.get('prior_tasks') or []
    non_ai_user = bool(data.get('non_ai_user', False))

    if batch_index >= _AI_MAX_BATCHES:
        return jsonify({'success': True, 'tasks': [], 'has_more': False})

    wrapper = get_session_wrapper(session_token)
    if not wrapper:
        return jsonify({'success': False, 'error': 'Invalid or expired session'}), 400

    iv = wrapper.interview_session
    job_description = _extract_job_description(iv)
    if not job_description:
        return jsonify({'success': False, 'error': 'No job description found'}), 400

    prior_block = ""
    if prior_tasks:
        prior_list = "\n".join(f"- {t}" for t in prior_tasks)
        prior_block = (
            f"\nTasks already shown (do NOT repeat or rephrase any of these):\n{prior_list}\n"
        )

    _BATCH_FOCUS = [
        # 0: new capabilities
        f"Generate the {_AI_BATCH_SIZE} most common new capabilities AI has given people in this role — "
        "things they can now do that weren't possible or practical before.",
        # 1: oversight / quality control
        f"Generate the {_AI_BATCH_SIZE} most common new oversight or quality-control responsibilities "
        "this role now has because AI is in the workflow — reviewing, verifying, or taking ownership of AI outputs.",
        # 2: writing & communication
        f"Generate {_AI_BATCH_SIZE} AI-related tasks specifically around writing, drafting, editing, or communicating — "
        "not yet listed above. Focus on how AI changes how people in this role produce or polish text.",
        # 3: analysis, research & decision support
        f"Generate {_AI_BATCH_SIZE} AI-related tasks specifically around data analysis, research, summarization, or decision-making — "
        "not yet listed above. Focus on how AI helps people in this role make sense of information.",
        # 4: automation, coding & workflow
        f"Generate {_AI_BATCH_SIZE} AI-related tasks specifically around automating repetitive work, writing code, or streamlining workflows — "
        "not yet listed above. Focus on how AI saves time or handles mechanical parts of the job.",
        # 5: learning, onboarding & skill-building
        f"Generate {_AI_BATCH_SIZE} AI-related tasks specifically around learning new things, onboarding others, training, or building skills — "
        "not yet listed above. Focus on how AI accelerates or supports knowledge transfer in this role.",
    ]
    if non_ai_user:
        _NON_AI_BATCH_FOCUS = [
            f"Generate the {_AI_BATCH_SIZE} most common oversight or quality-control responsibilities "
            "this role has taken on because AI is now in the workflow — reviewing, verifying, correcting, or taking ownership of AI-generated outputs from colleagues or systems.",
            f"Generate {_AI_BATCH_SIZE} responsibilities around adapting to AI-driven changes in this role — "
            "workflow changes, new expectations, or new accountabilities that emerged because others in the field adopted AI.",
            f"Generate {_AI_BATCH_SIZE} responsibilities around evaluating or gatekeeping AI outputs — "
            "deciding what AI-produced content is acceptable, flagging errors, or maintaining standards in an AI-augmented environment.",
            f"Generate {_AI_BATCH_SIZE} responsibilities around coordinating with or managing people who use AI — "
            "setting expectations, reviewing their AI-assisted work, or ensuring quality when collaborators use AI tools.",
            f"Generate {_AI_BATCH_SIZE} responsibilities around staying informed about AI's impact on this field — "
            "understanding what AI tools are being adopted, how they affect the work, or keeping skills current relative to AI-driven changes.",
        ]
        focus = _NON_AI_BATCH_FOCUS[min(batch_index, len(_NON_AI_BATCH_FOCUS) - 1)]
    else:
        focus = _BATCH_FOCUS[min(batch_index, len(_BATCH_FOCUS) - 1)]

    stop_hint = (
        "Set has_more to true unless you have genuinely exhausted every distinct AI-related oversight/governance area for this role."
        if non_ai_user else
        "Set has_more to true unless you have genuinely exhausted every distinct AI-related task area for this role "
        "across capability, oversight, tool use, quality control, and workflow integration. "
        "Err strongly on the side of true — there are almost always more areas to cover."
    )

    if non_ai_user:
        framing_rules = (
            "IMPORTANT: This participant said they do NOT personally use AI. "
            "Generate ONLY oversight/governance responsibilities — things that happen TO them or AROUND them because AI is in the environment.\n"
            "- Focus on: receiving AI-generated content from others, reviewing or approving AI outputs, "
            "adapting workflows because colleagues use AI, responding to AI-driven changes in their field, "
            "or managing deliverables that involve AI somewhere upstream.\n"
            "- Do NOT generate capability tasks. Do NOT frame tasks as 'Use AI to...', 'Generate with AI', or 'Ask AI to...'.\n"
            "- Good: 'Review AI-written drafts sent by collaborators', 'Adapt processes changed by team AI adoption'\n"
            "- Bad: 'Use AI to draft reports', 'Generate summaries with AI'\n"
            "- All tasks must have ai_type: 'governance'\n"
        )
    else:
        framing_rules = (
            "- The task NAME must make it explicit that AI is involved — include 'using AI', 'with AI', or a specific AI tool\n"
        )

    prompt = (
        f"Occupation context: \"{job_description}\"\n"
        f"{prior_block}\n"
        f"{focus}\n\n"
        "Rules:\n"
        f"{framing_rules}"
        "- Tasks should apply broadly to the occupational category, NOT be hyper-specific to this individual's niche\n"
        "  Good (researcher): 'Check AI-written summaries for errors before using them'\n"
        "  Too specific (researcher): 'Review AI outputs for bias in human-subjects consent forms'\n"
        "- Vary granularity: include a mix of broad tasks and more specific tasks. "
        "Do NOT make every task the same level of specificity.\n"
        "- Each task: 5–12 words, starts with an action verb, sentence case\n"
        "- Use plain, everyday language — no jargon or technical phrasing\n"
        "- Each needs a name and a one-sentence description (also plain language)\n"
        "- Label each with type: 'capability' or 'governance'\n"
        f"- {stop_hint}\n"
        "- Return ONLY valid JSON (no markdown fences):\n"
        '  {"tasks": [{"name": "...", "description": "...", "ai_type": "capability"}, ...], "has_more": true}'
    )

    try:
        engine = get_engine(model_name=_TASK_GEN_MODEL, temperature=1.0)
        response = invoke_engine(engine, prompt)
        raw = (response.content if hasattr(response, 'content') else str(response)).strip()
        parsed = _parse_tasks_json(raw)
        _AI_KEYWORDS = {"ai", "artificial intelligence", "chatgpt", "gpt", "claude",
                        "copilot", "gemini", "llm", "machine learning", "ml model",
                        "language model", "generative", "automation"}
        tasks = []
        for t in (parsed.get("tasks") or []):
            name = str(t.get("name", "")).strip()
            desc = str(t.get("description", "")).strip()
            ai_type = str(t.get("ai_type", "capability")).strip()
            if not name:
                continue
            name_lower = name.lower()
            if not any(kw in name_lower for kw in _AI_KEYWORDS):
                app.logger.warning(f"[ai_era_tasks] Dropping task with no AI mention: {name!r}")
                continue
            tasks.append({"name": name, "description": desc, "ai_type": ai_type})
        has_more = bool(parsed.get("has_more", batch_index + 1 < _AI_MAX_BATCHES))
        if batch_index + 1 >= _AI_MAX_BATCHES:
            has_more = False
    except Exception as e:
        app.logger.error(f"[ai_era_tasks] error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

    return jsonify({'success': True, 'tasks': tasks, 'has_more': has_more})


@app.route('/api/attention-check-tasks', methods=['POST'])
@agent_or_login_required
def attention_check_tasks():
    """Generate 2–3 tasks that are clearly unrelated to the user's profession (attention checks)."""
    from src.utils.llm.engines import get_engine, invoke_engine

    data = request.get_json(silent=True) or {}
    session_token = data.get('session_token')

    wrapper = get_session_wrapper(session_token)
    if not wrapper:
        return jsonify({'success': False, 'error': 'Invalid or expired session'}), 400

    iv = wrapper.interview_session
    job_description = _extract_job_description(iv)
    if not job_description:
        return jsonify({'success': False, 'error': 'No job description found'}), 400

    prompt = (
        f"Occupation context: \"{job_description}\"\n\n"
        "Generate exactly 8 work tasks that are CLEARLY AND OBVIOUSLY unrelated to this occupation — "
        "tasks someone in this role would never do as part of their job. "
        "These are attention-check items to verify the participant is reading carefully.\n\n"
        "Rules:\n"
        "- Each task must be unambiguously wrong for this profession (e.g. a software engineer would never 'Perform appendectomy surgery')\n"
        "- Still write them in the same format: 5–12 words, starts with an action verb, sentence case\n"
        "- Each needs a name (5–12 words) and a description (1 sentence describing what the task actually involves — write it as if it were a real task, same length and detail as any other task description)\n"
        "- Return ONLY valid JSON (no markdown fences):\n"
        '  {"tasks": [{"name": "Task name here", "description": "One sentence describing what this involves."}, ...]}'
    )

    try:
        engine = get_engine(model_name=_TASK_GEN_MODEL, temperature=0.9)
        response = invoke_engine(engine, prompt)
        raw = (response.content if hasattr(response, 'content') else str(response)).strip()
        parsed = _parse_tasks_json(raw)
        tasks = []
        for t in (parsed.get("tasks") or []):
            name = str(t.get("name", "")).strip()
            desc = str(t.get("description", "")).strip()
            if name:
                tasks.append({"name": name, "description": desc, "is_attention_check": True})
    except Exception as e:
        app.logger.error(f"[attention_check_tasks] error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

    return jsonify({'success': True, 'tasks': tasks})


@app.route('/api/ai-attention-check-tasks', methods=['POST'])
@agent_or_login_required
def ai_attention_check_tasks():
    """Generate 3 distractor tasks for the AI task widget — AI tasks from unrelated occupations that don't apply to this role."""
    from src.utils.llm.engines import get_engine, invoke_engine

    data = request.get_json(silent=True) or {}
    session_token = data.get('session_token')

    wrapper = get_session_wrapper(session_token)
    if not wrapper:
        return jsonify({'success': False, 'error': 'Invalid or expired session'}), 400

    iv = wrapper.interview_session
    job_description = _extract_job_description(iv)
    if not job_description:
        return jsonify({'success': False, 'error': 'No job description found'}), 400

    prompt = (
        f"Occupation context: \"{job_description}\"\n\n"
        "Generate exactly 3 AI-related tasks that someone in a COMPLETELY DIFFERENT occupation would do — "
        "tasks that involve AI but would make no sense for the role described above. "
        "These are attention-check items: the participant should NOT select them because they don't apply to their job.\n\n"
        "Rules:\n"
        "- Each task must come from an occupation clearly unrelated to the one described (e.g. if the role is a researcher, "
        "draw from roles like nurse, warehouse worker, retail manager, truck driver, chef, or construction supervisor)\n"
        "- Each task MUST involve AI — include 'using AI', 'with AI', or a specific AI tool in the name\n"
        "- The task must be plausible for that other occupation, but obviously irrelevant to the participant's actual role\n"
        "- Write them in the same format as the other tasks: 5–12 words, starts with an action verb, sentence case\n"
        "- Each needs a name and a one-sentence description\n"
        "- Return ONLY valid JSON (no markdown fences):\n"
        '  {"tasks": [{"name": "Task name here", "description": "One sentence describing what this involves."}, ...]}'
    )

    try:
        engine = get_engine(model_name=_TASK_GEN_MODEL, temperature=0.9)
        response = invoke_engine(engine, prompt)
        raw = (response.content if hasattr(response, 'content') else str(response)).strip()
        parsed = _parse_tasks_json(raw)
        tasks = []
        for t in (parsed.get("tasks") or []):
            name = str(t.get("name", "")).strip()
            desc = str(t.get("description", "")).strip()
            if name:
                tasks.append({"name": name, "description": desc, "is_attention_check": True})
    except Exception as e:
        app.logger.error(f"[ai_attention_check_tasks] error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

    return jsonify({'success': True, 'tasks': tasks})


@app.route('/api/regenerate-task-description', methods=['POST'])
@agent_or_login_required
def regenerate_task_description():
    """Return fresh one-sentence descriptions for one or more task names (batched)."""
    import re as _re
    from src.utils.llm.engines import get_engine, invoke_engine

    data = request.get_json(silent=True) or {}
    session_token = data.get('session_token')
    # Accept single task_name (legacy) or task_names list (batch)
    task_names_raw = data.get('task_names') or []
    single = data.get('task_name') or ''
    if single and not task_names_raw:
        task_names_raw = [single]
    task_names = [str(t).strip() for t in task_names_raw if str(t).strip()]

    if not task_names:
        return jsonify({'success': False, 'error': 'task_name or task_names required'}), 400

    wrapper = get_session_wrapper(session_token)
    if not wrapper:
        return jsonify({'success': False, 'error': 'Invalid or expired session'}), 400

    iv = wrapper.interview_session
    job_description = _extract_job_description(iv)

    if len(task_names) == 1:
        prompt = (
            f'Job context: "{job_description}"\n'
            f'Task name: "{task_names[0]}"\n\n'
            "Write exactly one sentence (under 20 words) describing what this task involves. "
            "Start with a verb. Return ONLY the sentence, no JSON, no quotes."
        )
        try:
            engine = get_engine(model_name=_TASK_GEN_MODEL, temperature=0.9)
            response = invoke_engine(engine, prompt)
            desc = (response.content if hasattr(response, 'content') else str(response)).strip()
            desc = _re.sub(r'^["\']|["\']$', '', desc).strip()
        except Exception as e:
            app.logger.error(f"[regenerate_task_description] error: {e}", exc_info=True)
            return jsonify({'success': False, 'error': 'Description generation failed'}), 500
        return jsonify({'success': True, 'description': desc, 'descriptions': {task_names[0]: desc}})

    # Batch: one LLM call for all names
    numbered = "\n".join(f"{i+1}. {name}" for i, name in enumerate(task_names))
    prompt = (
        f'Job context: "{job_description}"\n\n'
        f"For each task below, write exactly one sentence (under 20 words) describing what it involves. "
        f"Start each with a verb. Return ONLY a JSON object mapping each task name to its description:\n"
        f'{{"task name": "description sentence", ...}}\n\n'
        f"Tasks:\n{numbered}"
    )
    try:
        engine = get_engine(model_name=_TASK_GEN_MODEL, temperature=0.9)
        response = invoke_engine(engine, prompt)
        raw = (response.content if hasattr(response, 'content') else str(response)).strip()
        raw = _re.sub(r'^```[a-z]*\n?', '', raw)
        raw = _re.sub(r'\n?```$', '', raw).strip()
        parsed = json.loads(raw)
        descriptions = {str(k).strip(): str(v).strip() for k, v in parsed.items()}
        # Fill gaps with empty string for any name not returned
        for name in task_names:
            descriptions.setdefault(name, "")
    except Exception as e:
        app.logger.error(f"[regenerate_task_description] batch error: {e}", exc_info=True)
        descriptions = {name: "" for name in task_names}

    return jsonify({'success': True, 'descriptions': descriptions})


@app.route('/api/ai-widget-intro', methods=['POST'])
@agent_or_login_required
def ai_widget_intro():
    """Generate the intro message shown before the AI task widget.

    Reads the user's open-question answer and, if they disclaim AI use,
    returns a reframed message explaining indirect AI impact. Otherwise
    returns the default message.
    """
    from src.utils.llm.engines import get_engine, invoke_engine

    data = request.get_json(silent=True) or {}
    session_token = data.get('session_token')
    open_answer = str(data.get('open_answer') or '').strip()

    wrapper = get_session_wrapper(session_token)
    if not wrapper:
        return jsonify({'success': False, 'error': 'Invalid or expired session'}), 400

    iv = wrapper.interview_session
    job_description = _extract_job_description(iv)

    _DEFAULT = "Here are some AI-related tasks that may fit your work. Do any of these apply?"

    if not open_answer:
        return jsonify({'success': True, 'message': _DEFAULT})

    prompt = f"""A research participant was asked: "What are new things AI has enabled you to do, or new responsibilities you've taken on because of AI?"

Their answer: "{open_answer}"

Job context: {job_description or '(unknown)'}

Task: Decide whether the participant is disclaiming AI use (e.g., saying they don't use AI, haven't adopted it, or AI hasn't affected their work). If yes, write a single warm, conversational sentence that:
- Acknowledges their answer without repeating it back
- Explains that AI may still be affecting their work indirectly — e.g. because colleagues or collaborators use it, because their field is changing, or because they've had to adapt their work around AI tools others use
- Ends with "Do any of these apply?" so it leads naturally into a list of AI-related tasks

If the participant is NOT disclaiming AI use (i.e., they gave a positive or neutral answer), respond with exactly: DEFAULT

Respond with ONLY the sentence (or the word DEFAULT). No quotes, no explanation."""

    is_disclaimer = False
    try:
        engine = get_engine(model_name=_TASK_GEN_MODEL, temperature=0.7)
        response = invoke_engine(engine, prompt)
        raw = (response.content if hasattr(response, 'content') else str(response)).strip().strip('"\'')
        if raw == 'DEFAULT' or not raw:
            msg = _DEFAULT
        else:
            msg = raw
            is_disclaimer = True
    except Exception as e:
        app.logger.error(f"[ai_widget_intro] error: {e}", exc_info=True)
        msg = _DEFAULT

    print(f"[ai_widget_intro] is_ai_disclaimer={is_disclaimer}")
    return jsonify({'success': True, 'message': msg, 'is_ai_disclaimer': is_disclaimer})


@app.route('/api/task-followup', methods=['POST'])
@agent_or_login_required
def task_followup():
    """Return a brief interviewer response during the post-TVW probing phase.

    Detects whether the user is adding a new task, answering a clarifier, or
    signaling they are done. Returns {reply, done}. When done=true during
    ai_extras, the session is ended server-side.
    """
    from src.utils.llm.engines import get_engine, invoke_engine

    data = request.get_json(silent=True) or {}
    session_token = data.get('session_token')
    task_text = (data.get('task_text') or '').strip()
    prior_tasks = data.get('prior_tasks') or []
    phase = _normalize_task_followup_phase(data.get('phase') or 'probing')
    recent_dialogue = _normalize_task_followup_dialogue(data.get('recent_dialogue'))

    if not task_text:
        return jsonify({'success': False, 'error': 'task_text required'}), 400

    wrapper = get_session_wrapper(session_token)
    if not wrapper:
        return jsonify({'success': False, 'error': 'Invalid or expired session'}), 400

    iv = wrapper.interview_session
    job_description = _extract_job_description(iv)

    # ── ai_open phase: brief clarifying conversation before the AI task widget ──
    if phase == 'ai_open':
        history = _task_followup_history(session_token, phase)
        if recent_dialogue:
            history[:] = recent_dialogue[-_TASK_FOLLOWUP_HISTORY_MAX_MESSAGES:]

        history_block = _format_task_followup_history(history)
        turn_count = sum(1 for h in history if h['role'] == 'Participant')

        prompt = (
            f"You are a warm, concise interviewer. You just asked the participant:\n"
            f"\"With AI becoming part of so many workflows, what are new things AI has enabled you to do, "
            f"or new responsibilities you've taken on because of AI?\"\n\n"
            f"Occupation context: \"{job_description}\"\n\n"
            f"Conversation so far (oldest to newest):\n{history_block}\n\n"
            f"Participant's latest message: \"{task_text}\"\n\n"
            f"Number of participant turns so far (including this one): {turn_count + 1}\n\n"
            "Decide which case applies:\n\n"
            "CASE A — enough context gathered, move on:\n"
            "  Apply this when ANY of these are true:\n"
            "  - The participant named at least one specific AI-related activity (e.g. a tool, a use case, a task) AND this is turn 2 or later.\n"
            "  - This is turn 3 or later, regardless of answer quality.\n"
            "  - The participant said they don't use AI or AI hasn't changed their work.\n"
            "  If the participant said they DON'T use AI or AI hasn't affected their work, return an EMPTY reply (reply: \"\") — a follow-up message will handle this case.\n"
            "  Otherwise give a brief, warm acknowledgment (1 sentence, no question).\n"
            "  Return JSON: {\"reply\": \"...\", \"done\": true}\n\n"
            "CASE B — need one clarifying question:\n"
            "  Apply this only when: the answer is too vague to understand what AI role it plays "
            "(e.g. 'a lot more', 'I use it sometimes', 'yes definitely') AND this is turn 1.\n"
            "  Ask ONE short, specific follow-up that anchors on a concrete detail they mentioned. "
            "Do not ask generic questions like 'can you tell me more?'.\n"
            "  Return JSON: {\"reply\": \"...\", \"done\": false}\n\n"
            "Rules:\n"
            "- reply: 1 sentence max; at most one question mark\n"
            "- Do NOT use em dashes or en dashes; use plain punctuation\n"
            "- Return ONLY valid JSON, no markdown fences, no extra text"
        )

        done_flag = False
        reply = ""
        try:
            engine = get_engine(model_name=_TASK_GEN_MODEL, temperature=0.7)
            response = invoke_engine(engine, prompt)
            raw = (response.content if hasattr(response, 'content') else str(response)).strip()
            if raw.startswith("```"):
                raw = raw[raw.find("\n") + 1:] if "\n" in raw else raw.lstrip("`")
            raw = raw.rstrip("`").strip()
            parsed = json.loads(raw)
            reply = str(parsed.get('reply', '')).strip()
            done_flag = bool(parsed.get('done', False))
        except Exception as e:
            app.logger.error(f"[task_followup/ai_open] error: {e}", exc_info=True)
            done_flag = True

        reply = reply.replace("—", ",").replace("–", ",")
        reply = " ".join(reply.split())

        history.append({"role": "Participant", "content": task_text})
        if reply:
            history.append({"role": "Interviewer", "content": reply})
        if len(history) > _TASK_FOLLOWUP_HISTORY_MAX_MESSAGES:
            del history[:-_TASK_FOLLOWUP_HISTORY_MAX_MESSAGES]

        open_answer = " ".join(h['content'] for h in history if h['role'] == 'Participant')
        return jsonify({'success': True, 'reply': reply, 'done': done_flag, 'open_answer': open_answer})

    is_ai_extras = phase == 'ai_extras'
    inventory_scope = "an AI-related task inventory" if is_ai_extras else "a task inventory"
    remaining_goal = (
        "capture remaining AI-related tasks only, meaning new things AI enables them to do "
        "or new responsibilities around reviewing, verifying, or overseeing AI outputs"
        if is_ai_extras
        else "capture remaining tasks"
    )
    collected_label = "AI-related tasks already collected" if is_ai_extras else "Tasks already collected"
    other_tasks_question = (
        "Any other AI-related tasks come to mind?"
        if is_ai_extras
        else "What other tasks are part of your work?"
    )
    new_task_scope = (
        "participant is adding a NEW AI-related task (describes work involving AI tools, AI outputs, or AI oversight)"
        if is_ai_extras
        else "participant is adding a NEW task (describes distinct work they do)"
    )
    scope_rule = (
        "- This phase is ONLY about AI-related tasks. Do NOT broaden to general work tasks or ask what other tasks take up their time.\n"
        "- When asking for more, vary the wording and keep it brief. Do not repeat the exact same 'other AI-related tasks' sentence after every answer.\n"
        if is_ai_extras
        else ""
    )
    case_c_instruction = (
        "  Respond briefly to the answer, then ask a short, varied continuation question scoped to AI-related tasks. "
        "Examples: 'Anything else AI-related come to mind?', 'Any other AI-enabled work or AI oversight tasks?', "
        "'Are there other ways AI shows up in your work?'. Do NOT reuse the same wording from the previous interviewer turn.\n"
        if is_ai_extras
        else "  Respond in 1 sentence that briefly reflects understanding, then ask a short, varied continuation question "
        "scoped to their work tasks. Examples: 'What other tasks are part of your work?', 'What else takes up your time?', "
        "'Are there other things you regularly work on?'. Do NOT reuse the same wording from the previous interviewer turn.\n"
    )

    history = _task_followup_history(session_token, phase)
    if recent_dialogue:
        history[:] = recent_dialogue[-_TASK_FOLLOWUP_HISTORY_MAX_MESSAGES:]
    elif not history:
        seed = other_tasks_question if is_ai_extras else _latest_interviewer_message_from_chat(iv)
        if seed:
            history.append({"role": "Interviewer", "content": seed})

    history_block = _format_task_followup_history(history)

    # Track follow-up turns per task so we can cap at 2
    followup_turns_bucket = task_followup_turns_by_session.setdefault(session_token, {})
    followup_turns = followup_turns_bucket.get(phase, 0)
    followup_limit_note = (
        f"IMPORTANT: This is follow-up turn {followup_turns + 1} for the current task. "
        f"You may ask at most 2 follow-up turns per task. "
        f"{'You have reached the limit — classify as CASE C and immediately redirect to asking what else they do.' if followup_turns >= 2 else 'After 2 follow-up turns, redirect to asking what else.'}\n\n"
    )

    prior_block = ""
    if prior_tasks:
        prior_block = f"{collected_label}:\n" + "\n".join(f"- {t}" for t in prior_tasks) + "\n\n"

    prompt = (
        f"You are a warm, concise interviewer collecting {inventory_scope}.\n"
        f"Goal: {remaining_goal} while sounding like you understood what the participant just said.\n\n"
        f"Occupation context: \"{job_description}\"\n"
        f"{prior_block}"
        f"Recent task-followup conversation (oldest to newest):\n{history_block}\n\n"
        f"Participant's latest message: \"{task_text}\"\n\n"
        f"{followup_limit_note}"
        "Decide which case applies:\n\n"
        "CASE A — participant is DONE adding tasks:\n"
        "  Only classify as DONE for explicit done signals: 'no', 'nope', 'that's it', "
        "'nothing else', 'I think that's everything', 'all good', 'nothing more', 'no more', "
        "'can't think of anything', or a short affirmation clearly answering a prior 'any other tasks?' question.\n"
        "  CRITICAL: if the message describes actual work, tasks, or activities — even briefly — classify as CASE B, not CASE A.\n"
        + (
        "  Return JSON: {\"reply\": \"\", \"done\": true}\n\n"
        if not is_ai_extras else
        "  Respond warmly in 1 sentence. Do NOT reference any specific task or topic they mentioned. "
        "  Keep it general — e.g. 'Thank you for giving me an in-depth look at your work' "
        "  Return JSON: {\"reply\": \"...\", \"done\": true}\n\n"
        )
        + f"CASE B — {new_task_scope}:\n"
        "  Ask exactly one follow-up question about that task.\n"
        "  The question must show understanding by anchoring to a concrete detail from their latest message.\n"
        "  Do NOT use generic filler phrases like 'Got it', 'Good one', 'Thanks for sharing', or 'Noted'.\n"
        "  Do NOT end with 'anything else?' in this case.\n"
        "  Also extract a concise task name in 'Action + Object + Objective' format — a verb phrase "
        "  that captures what they do, what they do it to, and why/to what end "
        "  (e.g. 'Attend CS seminars to stay current with research trends', "
        "  'Analyze experimental results to validate hypotheses', "
        "  'Write grant proposals to secure project funding'). "
        "  Return JSON: {\"reply\": \"...\", \"task_name\": \"...\", \"done\": false}\n\n"
        "CASE C — participant is answering your previous follow-up/clarifier (including short replies like "
        "'both', 'yes', 'correct', or noun phrases like 'experimental results') and did NOT introduce a new task:\n"
        f"{case_c_instruction}"
        "  Return JSON: {\"reply\": \"...\", \"done\": false}\n\n"
        "Rules:\n"
        f"{scope_rule}"
        "- reply: 1 sentence max; at most one question mark\n"
        "- task_name: starts with a verb, includes action + object + objective (purpose/outcome), aim for 8–12 words\n"
        "- For CASE B, follow-up must be specific to this participant's latest message and show understanding\n"
        "- Do NOT repeat the task name verbatim in the reply\n"
        "- Do NOT use em dashes or en dashes in the reply; use plain punctuation\n"
        "- Return ONLY valid JSON, no markdown fences, no extra text"
    )

    done = False
    reply = other_tasks_question
    task_name = None
    try:
        engine = get_engine(model_name=_TASK_GEN_MODEL, temperature=0.9)
        response = invoke_engine(engine, prompt)
        raw = (response.content if hasattr(response, 'content') else str(response)).strip()
        if raw.startswith("```"):
            first_newline = raw.find("\n")
            if first_newline != -1:
                raw = raw[first_newline + 1 :]
            else:
                raw = raw.lstrip("`")
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
        parsed = json.loads(raw)
        reply = str(parsed.get('reply', reply)).strip()
        done = bool(parsed.get('done', False))
        if not done and parsed.get('task_name'):
            task_name = str(parsed['task_name']).strip()
    except Exception as e:
        app.logger.error(f"[task_followup] error: {e}", exc_info=True)

    # Keep replies punctuation-simple for frontend readability.
    reply = reply.replace("—", ",").replace("–", ",")
    reply = " ".join(reply.split())

    # Ensure non-done responses continue the probing flow with one question.
    if not done and "?" not in reply:
        reply = f"{reply.rstrip('. ')}. {other_tasks_question}"

    # Update follow-up turn counter: reset on new task, increment on follow-up
    if task_name:
        followup_turns_bucket[phase] = 0
    elif not done:
        followup_turns_bucket[phase] = followup_turns + 1

    history.append({"role": "Participant", "content": task_text})
    if reply:
        history.append({"role": "Interviewer", "content": reply})
    if len(history) > _TASK_FOLLOWUP_HISTORY_MAX_MESSAGES:
        del history[:-_TASK_FOLLOWUP_HISTORY_MAX_MESSAGES]

    if done and phase == 'ai_extras':
        def _end():
            iv.end_with_thankyou(send_message=True)
        if hasattr(wrapper, 'loop'):
            wrapper.loop.call_soon_threadsafe(_end)
        else:
            _end()
        # End of post-task flow; clear rolling state.
        task_followup_history_by_session.pop(session_token, None)
        task_followup_turns_by_session.pop(session_token, None)

    return jsonify({'success': True, 'reply': reply, 'task_name': task_name, 'done': done})


@app.route('/api/submit-task-validation', methods=['POST'])
@agent_or_login_required
def submit_task_validation():
    """Store the validated task list and trigger session end (feedback widget)."""
    data = request.get_json(silent=True) or {}
    session_token = data.get('session_token')
    validated_tasks = data.get('tasks') or []
    extra_tasks = data.get('extra_tasks') or []
    wishlist_tasks = data.get('wishlist_tasks') or []
    attn_failed = int(data.get('attn_failed', 0))
    attn_total = int(data.get('attn_total', 0))

    if not isinstance(validated_tasks, list):
        return jsonify({'success': False, 'error': 'tasks must be a list'}), 400

    wrapper = get_session_wrapper(session_token)
    if not wrapper:
        return jsonify({'success': False, 'error': 'Invalid or expired session'}), 400

    iv = wrapper.interview_session
    # Reset post-validation probing dialogue state for this session.
    task_followup_history_by_session.pop(session_token, None)
    task_followup_turns_by_session.pop(session_token, None)
    all_tasks = [str(t).strip() for t in validated_tasks + extra_tasks if str(t).strip()]
    wishlist = [str(t).strip() for t in wishlist_tasks if str(t).strip()]

    # Persist tasks into the user portrait
    agenda = getattr(iv, 'session_agenda', None)
    if agenda is not None:
        portrait = dict(agenda.user_portrait or {})
        portrait['Task Inventory'] = all_tasks
        if wishlist:
            portrait['Task Wishlist'] = wishlist
        portrait = normalize_user_portrait(portrait)
        agenda.user_portrait = portrait
        portrait_path = os.path.join(
            os.getenv('LOGS_DIR', 'logs'), iv.user_id, 'user_portrait.json'
        )
        os.makedirs(os.path.dirname(portrait_path), exist_ok=True)
        with open(portrait_path, 'w') as f:
            json.dump(portrait, f, indent=2)
        app.logger.info(
            f"[submit_task_validation] Saved {len(all_tasks)} tasks for user {iv.user_id}"
        )

    # Persist attention check result to user record
    attn_disqualified = False
    attn_max = int(os.getenv('ATTN_CHECK_MAX_FAILS', '1'))
    if attn_total > 0:
        users = load_users()
        if iv.user_id in users:
            users[iv.user_id]['attn_failed'] = attn_failed
            users[iv.user_id]['attn_total'] = attn_total
            save_users(users)
        app.logger.info(
            f"[submit_task_validation] attn {attn_failed}/{attn_total} failed "
            f"(max allowed: {attn_max}) for user {iv.user_id}"
        )
        if attn_failed > attn_max:
            attn_disqualified = True

    if attn_disqualified:
        # End the session now — skip probing loop and AI tasks
        def _end_early():
            iv.end_with_thankyou(send_message=False)
        if hasattr(wrapper, 'loop'):
            wrapper.loop.call_soon_threadsafe(_end_early)
        else:
            _end_early()
        return jsonify({'success': True, 'task_count': len(all_tasks), 'attn_disqualified': True})

    # Ask the participant if anything was missed before ending
    def _inject_question():
        iv.add_message_to_chat_history(
            role="Interviewer",
            content="Are there any tasks you can think of that we haven't covered?",
            message_type="conversation",
        )

    if hasattr(wrapper, 'loop'):
        wrapper.loop.call_soon_threadsafe(_inject_question)
    else:
        _inject_question()

    return jsonify({'success': True, 'task_count': len(all_tasks)})


# =============================================================================
# HEALTH CHECK - NOT PROTECTED (for monitoring)
# =============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint - no login required for monitoring"""
    current_time = time.time()
    session_ages = []
    for wrapper in active_sessions.values():
        age_minutes = (current_time - wrapper.created_at) / 60
        session_ages.append(age_minutes)
    
    avg_age = sum(session_ages) / len(session_ages) if session_ages else 0
    
    return jsonify({
        'status': 'healthy',
        'active_sessions': len(active_sessions),
        'avg_session_age_minutes': round(avg_age, 2),
        'tts_provider': TTS_PROVIDER,
        'tts_voice': TTS_VOICE,
        'uptime_seconds': round(current_time - START_TIME, 2)
    })

# =============================================================================
# VOICE PROCESSING UTILITIES
# =============================================================================

def generate_speech_from_text(text: str, output_path: Path) -> Path:
    """Generate speech audio from text using TTS"""
    global tts_engine
    if tts_engine is None:
        raise NotImplementedError("TTS engine is not configured")
    os.makedirs(os.path.dirname(str(output_path)) or '.', exist_ok=True)
    result_path = tts_engine.text_to_speech(text=text, output_path=str(output_path))
    return Path(result_path)

def transcribe_audio_to_text(audio_path: Path) -> str:
    """Transcribe audio file to text using speech recognition"""
    global stt_engine
    if stt_engine is None:
        raise RuntimeError("STT engine is not initialized. Check speech_to_text setup.")
    return stt_engine.transcribe(str(audio_path))


# =============================================================================
# SESSION CLEANUP
# =============================================================================

def cleanup_old_sessions():
    """Remove sessions older than timeout threshold and clean stale audio cache"""
    current_time = time.time()
    to_remove = []

    for token, wrapper in list(active_sessions.items()):
        age = current_time - wrapper.created_at
        if age > SESSION_TIMEOUT_SECONDS:
            to_remove.append(token)
            session_audio_cache.pop(token, None)
            task_followup_history_by_session.pop(token, None)
            task_followup_turns_by_session.pop(token, None)
            pending_turns_by_session.pop(token, None)
            delivered_turn_messages_by_session.pop(token, None)

    for token in to_remove:
        wrapper = active_sessions.pop(token, None)
        if wrapper:
            print(f"[Cleanup] Removed session {token} (age: {age/60:.1f}min, user: {wrapper.user_id})")

    if to_remove:
        print(f"[Cleanup] Removed {len(to_remove)} old sessions. Active: {len(active_sessions)}")

    # Clean up stale pending audio cache entries (older than 5 minutes)
    audio_cleaned = 0
    for session_token, cache in list(session_audio_cache.items()):
        for message_id, entry in list(cache.items()):
            if isinstance(entry, dict):
                entry_age = current_time - entry.get('timestamp', current_time)
                # Remove pending entries older than 5 minutes (likely failed)
                if entry.get('status') == 'pending' and entry_age > 300:
                    cache.pop(message_id, None)
                    audio_cleaned += 1
                # Remove failed entries older than 1 hour
                elif entry.get('status') == 'failed' and entry_age > 3600:
                    cache.pop(message_id, None)
                    audio_cleaned += 1

    if audio_cleaned > 0:
        print(f"[Cleanup] Removed {audio_cleaned} stale audio cache entries")

def start_cleanup_thread():
    """Start background thread for session cleanup"""
    def cleanup_loop():
        while True:
            time.sleep(300)  # Every 5 minutes
            try:
                cleanup_old_sessions()
            except Exception as e:
                print(f"[Cleanup] Error: {e}")
    
    t = threading.Thread(target=cleanup_loop, daemon=True, name="SessionCleanup")
    t.start()
    print("[Cleanup] Started session cleanup thread")

# =============================================================================
# MAIN
# =============================================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Flask Interview Session Web Application'
    )
    parser.add_argument('--user-id', type=str, help='Default user ID')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--additional_context_path', default=None)
    parser.add_argument('--restart', action='store_true', default=False)
    parser.add_argument('--max_turns', type=int, default=None)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    if args.restart and args.user_id:
        os.system(f"rm -rf {os.getenv('LOGS_DIR')}/{args.user_id}")
        os.system(f"rm -rf {os.getenv('DATA_DIR')}/{args.user_id}")
        print(f"Cleared data for user {args.user_id}")
    
    config.default_user_id = args.user_id if args.user_id else "web_guest"
    config.host = args.host
    config.port = args.port
    config.debug = args.debug
    config.restart = args.restart
    config.max_turns = args.max_turns
    config.additional_context_path = args.additional_context_path
    
    start_cleanup_thread()
    
    print("\n" + "="*70)
    print("Flask Interview Session Server - Multi-User Mode")
    print("="*70)
    print(f"🌐 Host:              {config.host}")
    print(f"🔌 Port:              {config.port}")
    print(f"🐛 Debug:             {config.debug}")
    print(f"🔐 Authentication:    Enabled")
    print(f"🧹 Session Cleanup:   Every 5 minutes (timeout: {SESSION_TIMEOUT_SECONDS/60:.0f} min)")
    print(f"🗣️  TTS Provider:      {TTS_PROVIDER} ({TTS_VOICE})")
    print("="*70)
    print(f"\n📍 Login at: http://{config.host}:{config.port}/login")
    print(f"📊 Health check: http://{config.host}:{config.port}/health")
    print("="*70 + "\n")
    
    if not config.debug:
        print("⚠️  For production, use: gunicorn -w 2 --threads 4 -b 0.0.0.0:8080 flask_app:app\n")
    
    app.run(
        host=config.host,
        port=config.port,
        debug=config.debug,
        use_reloader=False,
        threaded=True
    )
