from src.utils.llm import engines


def test_get_engine_supports_gpt_5_4(monkeypatch):
    captured = {}

    def fake_ctor(**kwargs):
        captured.update(kwargs)
        return {"engine": "fake"}

    monkeypatch.setitem(engines.engine_constructor, "gpt-5.4", fake_ctor)

    out = engines.get_engine("gpt-5.4", max_tokens=1234, temperature=0.0)

    assert out == {"engine": "fake"}
    assert captured["model_name"] == "gpt-5.4"
    assert captured["max_completion_tokens"] == 1234
    # gpt-5 models force the default temperature.
    assert captured["temperature"] == 1
    assert "max_tokens" not in captured

