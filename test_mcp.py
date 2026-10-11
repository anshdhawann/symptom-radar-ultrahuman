"""The stdio boundary must answer bad input and keep serving."""
import io
import json

import pytest

from symptomradar import mcp, report, storage


def request(method="tools/call", **params):
    return {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}


def serve(monkeypatch, bad):
    raw = bad if isinstance(bad, str) else json.dumps(bad)
    source = io.StringIO(raw + '\n' + json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"}) + '\n')
    output = io.StringIO()
    monkeypatch.setattr(mcp.sys, "stdin", source)
    monkeypatch.setattr(mcp.sys, "stdout", output)
    mcp.run_mcp_server()
    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert responses[-1] == {"jsonrpc": "2.0", "id": 2, "result": {}}
    return responses[:-1]


@pytest.mark.parametrize("bad, code", [
    ('{', -32700),
    ([], -32600), (None, -32600), (1, -32600),
    ({"jsonrpc": "1.0", "id": 1, "method": "ping"}, -32600),
    ({"jsonrpc": "2.0", "id": 1, "method": []}, -32600),
    ({"jsonrpc": "2.0", "id": [], "method": "ping"}, -32600),
    ({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": []}, -32602),
    ({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": None}, -32602),
    (request(name="symptom_radar_history", arguments=[]), -32602),
    (request(name="symptom_radar_history", arguments="bad"), -32602),
    (request(name="symptom_radar_history", arguments=None), -32602),
    (request(name="unknown"), -32601), (request(method="unknown"), -32601),
    *[(request(name="symptom_radar_history", arguments={"days": days}), -32602)
      for days in ("3", 1.5, True, None, -1, 0, 10**100)],
    (request(name="symptom_radar_history", arguments={"metrics": "date"}), -32602),
    (request(name="symptom_radar_history", arguments={"metrics": [[]]}), -32602),
    (request(name="symptom_radar_label", arguments={"label": "invalid"}), -32602),
])
def test_mcp_error_and_continuation(monkeypatch, tmp_path, bad, code):
    monkeypatch.setattr(storage, "DB_PATH", str(tmp_path / "mcp.db"))
    responses = serve(monkeypatch, bad)
    assert len(responses) == 1
    assert responses[0]["jsonrpc"] == "2.0"
    assert responses[0]["error"]["code"] == code


def test_mcp_internal_exception_and_continuation(monkeypatch):
    def broken():
        raise RuntimeError("private tool detail")
    monkeypatch.setattr(report, "build_report", broken)
    responses = serve(monkeypatch, request(name="symptom_radar_report"))
    assert responses == [{"jsonrpc": "2.0", "id": 1,
                          "error": {"code": -32603, "message": "Internal error"}}]


@pytest.mark.parametrize("notification", [
    {"jsonrpc": "2.0", "method": "notifications/initialized"},
    {"jsonrpc": "2.0", "method": "unknown"},
    {"jsonrpc": "2.0", "method": "tools/call", "params": []},
])
def test_mcp_notifications_are_silent(monkeypatch, notification):
    assert serve(monkeypatch, notification) == []


@pytest.mark.parametrize("days", [1, 90])
def test_mcp_valid_days_and_null_id(monkeypatch, tmp_path, days):
    monkeypatch.setattr(storage, "DB_PATH", str(tmp_path / "mcp.db"))
    req = request(name="symptom_radar_history", arguments={"days": days})
    req["id"] = None
    responses = serve(monkeypatch, req)
    assert responses == [{"jsonrpc": "2.0", "id": None,
                          "result": {"content": [{"type": "text", "text": "[]"}]}}]
