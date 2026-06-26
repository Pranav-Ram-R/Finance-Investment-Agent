from finplan.tools.news import get_news_sentiment, score_headlines


def test_positive_headlines_read_positive():
    r = score_headlines([
        "Nifty surges to record high as banks rally",
        "Sensex jumps; strong profit growth boosts sentiment",
    ])
    assert r["label"] == "positive"
    assert r["score"] > 0
    assert r["positive_hits"] > r["negative_hits"]


def test_negative_headlines_read_negative():
    r = score_headlines([
        "Markets plunge as recession fears mount",
        "Nifty falls; weak earnings trigger selloff",
    ])
    assert r["label"] == "negative"
    assert r["score"] < 0


def test_no_headlines_is_neutral():
    r = score_headlines([])
    assert r["label"] == "neutral"
    assert r["score"] == 0.0


def test_fetch_degrades_gracefully_without_network(monkeypatch):
    # If the feed returns nothing, the tool must not crash — it reports no news.
    import finplan.tools.news as news

    class _Dummy:
        news = []

    monkeypatch.setattr(news.yf, "Ticker", lambda _t: _Dummy())
    out = get_news_sentiment("^NSEI")
    assert out["status"] == "no_recent_news"
    assert out["label"] == "unavailable"
