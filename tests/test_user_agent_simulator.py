from collections import deque

from src.agents.user.user_agent import UserAgent


def _make_agent(threshold: float = 0.7) -> UserAgent:
    agent = UserAgent.__new__(UserAgent)
    agent.reuse_similar_answers = True
    agent.similar_question_threshold = threshold
    agent._qa_history = deque(
        [
            {
                "question": "Can you walk me through a typical day at work?",
                "answer": "I start by triaging support tickets and then work on bug fixes.",
            },
            {
                "question": "What tools do you use most often?",
                "answer": "I mostly use Python, SQL, and internal dashboards.",
            },
        ],
        maxlen=200,
    )
    return agent


def test_find_similar_answer_reuses_prior_response():
    agent = _make_agent(threshold=0.55)
    result = agent._find_similar_answer(
        "Could you walk me through your typical day at work?"
    )
    assert result is not None
    answer, score, matched_question = result
    assert "triaging support tickets" in answer
    assert matched_question == "Can you walk me through a typical day at work?"
    assert score >= 0.55


def test_find_similar_answer_returns_none_when_different():
    agent = _make_agent(threshold=0.7)
    result = agent._find_similar_answer("How do you prefer to receive feedback?")
    assert result is None


def test_question_similarity_is_exact_match_for_normalized_duplicates():
    agent = _make_agent()
    score = agent._question_similarity(
        "What's your role??",
        "whats your role",
    )
    assert score == 1.0


def test_extract_interviewer_user_pairs_merges_consecutive_user_turns():
    messages = [
        {"speaker": "Interviewer", "content": "What do you do weekly?"},
        {"speaker": "User", "content": "Read papers."},
        {"speaker": "User", "content": "Write code."},
        {"speaker": "Interviewer", "content": "Any occasional tasks?"},
        {"speaker": "User", "content": "Review thesis sections."},
    ]
    pairs = UserAgent._extract_interviewer_user_pairs(messages)
    assert pairs == [
        {"question": "What do you do weekly?", "answer": "Read papers.\nWrite code."},
        {"question": "Any occasional tasks?", "answer": "Review thesis sections."},
    ]


def test_seed_from_original_transcript_populates_qa_history(tmp_path, monkeypatch):
    user_id = "sim_user_123"
    profiles_dir = tmp_path / "profiles"
    user_dir = profiles_dir / user_id
    user_dir.mkdir(parents=True, exist_ok=True)

    transcript = tmp_path / "source_chat_history.log"
    transcript.write_text(
        "\n".join(
            [
                "2026-04-14 09:00:00,000 - INFO - Interviewer: Main task?",
                "2026-04-14 09:00:01,000 - INFO - User: Writing code to analyze data.",
                "2026-04-14 09:00:02,000 - INFO - Interviewer: Next task?",
                "2026-04-14 09:00:03,000 - INFO - User: Writing papers.",
            ]
        ),
        encoding="utf-8",
    )

    artifact = user_dir / f"{user_id}_derived_tasks_from_transcript.json"
    artifact.write_text(
        '{"source_chat_history_log": "' + str(transcript).replace("\\", "\\\\") + '"}',
        encoding="utf-8",
    )

    monkeypatch.setenv("USER_AGENT_PROFILES_DIR", str(profiles_dir))

    agent = UserAgent.__new__(UserAgent)
    agent.user_id = user_id
    agent.transcript_anchor_max_pairs = 10
    agent._qa_history = deque(maxlen=50)

    seeded = agent._seed_from_original_transcript()
    assert seeded == 2
    assert len(agent._qa_history) == 2
    assert agent._qa_history[0]["question"] == "Main task?"
    assert "Writing code" in agent._qa_history[0]["answer"]
