import src.main_flask as main_flask


def test_available_time_accepts_arbitrary_minutes():
    assert main_flask._normalize_available_time_minutes("17") == 17
    assert main_flask._normalize_available_time_minutes(23) == 23


def test_available_time_is_bounded():
    assert main_flask._normalize_available_time_minutes("2") == 5
    assert main_flask._normalize_available_time_minutes("999") == 120


def test_available_time_uses_fallback_for_invalid_value():
    assert main_flask._normalize_available_time_minutes("soon", fallback=20) == 20
