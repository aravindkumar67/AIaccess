from fallwatch.api import EventApi

def test_disabled_api_no_network():
    assert EventApi(enabled=False).emit({"event_id":"test"})=={"sent":False,"reason":"disabled"}

def test_enabled_without_url_reports_configuration_error():
    assert EventApi(enabled=True,url="").emit({"event_id":"test"})["reason"]=="missing_url"
