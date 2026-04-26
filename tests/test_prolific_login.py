import json
from pathlib import Path

import pytest

import src.main_flask as main_flask


@pytest.fixture
def client(tmp_path, monkeypatch):
    users_file = tmp_path / "users.json"
    monkeypatch.setattr(main_flask, "USERS_FILE", str(users_file))
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    main_flask.app.config["TESTING"] = True

    with main_flask.app.test_client() as test_client:
        yield test_client, users_file, tmp_path


def test_prolific_get_auto_logs_in_without_random_id_screen(client):
    test_client, users_file, tmp_path = client

    res = test_client.get(
        "/login?PROLIFIC_PID=PID123ABC&STUDY_ID=study-1&SESSION_ID=session-1"
    )

    assert res.status_code == 302
    assert res.headers["Location"].endswith("/?show_intro=1")

    users = json.loads(users_file.read_text())
    assert "PID123ABC" in users
    assert users["PID123ABC"]["username"] == "PID123ABC"
    assert users["PID123ABC"]["prolific_pid"] == "PID123ABC"
    assert users["PID123ABC"]["prolific_study_id"] == "study-1"
    assert users["PID123ABC"]["prolific_session_id"] == "session-1"
    assert (tmp_path / "logs" / "PID123ABC").is_dir()
    assert (tmp_path / "data" / "PID123ABC").is_dir()


def test_non_prolific_get_still_shows_random_id_screen(client):
    test_client, _, _ = client

    res = test_client.get("/login")
    body = res.get_data(as_text=True)

    assert res.status_code == 200
    assert "Continue with a random ID" in body
    assert "id=\"username-input\"" in body


def test_prolific_post_uses_pid_as_user_id_and_username(client):
    test_client, users_file, _ = client

    res = test_client.post(
        "/login",
        data={
            "username": "should-not-use-random",
            "prolific_pid": "PID456DEF",
            "study_id": "study-2",
            "prolific_session_id": "session-2",
        },
    )

    assert res.status_code == 302
    users = json.loads(users_file.read_text())
    assert "PID456DEF" in users
    assert users["PID456DEF"]["username"] == "PID456DEF"
    assert users["PID456DEF"]["prolific_pid"] == "PID456DEF"
