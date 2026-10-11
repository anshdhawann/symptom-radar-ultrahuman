from . import config, engine, report, storage
import json, sys

class RpcError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _mcp_handle(method, params):
    """Dispatch one MCP method call. Returns (result_dict, error_str_or_None)."""
    if method == "initialize":
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {
                "name": "symptom-radar",
                "version": "1.0.0",
            },
        }, None

    if method == "tools/list":
        return {"tools": [
            {
                "name": "symptom_radar_report",
                "description": "Fetch today's Ultrahuman data, store a snapshot, and return the daily report (markdown) including the Symptom Radar strain assessment.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "symptom_radar_history",
                "description": "Query the local biometric database. Returns the most recent N days of one or more metrics, oldest-first. Metrics: sleep_score, total_sleep_min, night_rhr, sleep_rhr, avg_sleep_hrv, temp_deviation, recovery_index, movement_index, total_steps, vo2_max, hr_drop, morning_alertness, spo2.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "days": {"type": "integer", "default": 14, "minimum": 1, "maximum": 90},
                        "metrics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Columns to return. Omit for a default set (date, sleep_score, night_rhr, sleep_rhr, avg_sleep_hrv, temp_deviation).",
                        },
                    },
                },
            },
            {
                "name": "symptom_radar_strain",
                "description": "Return today's Symptom Radar strain level (0/1/2), the strain index, and the per-metric breakdown without re-fetching from the API.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "symptom_radar_label",
                "description": "Log a daily self-report label — the training signal that lets the engine distinguish real illness from alcohol strain / poor sleep (both look identical in the biometrics). Call this daily with how the user felt.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "enum": ["fine", "rough", "sick"],
                                  "description": "fine=normal, rough=hangover/tired/stressed, sick=actually felt ill"},
                        "date": {"type": "string", "description": "YYYY-MM-DD (defaults to today in the wearer timezone)"},
                        "note": {"type": "string", "description": "Optional context, e.g. 'hangover' or 'fever'"},
                        "confounders": {"type": "string", "description": "alcohol=3,exercise,travel,sleep_loss; use =false or alcohol=0 for explicit absence. exercise means hard exercise."},
                        "alcohol": {"type": "integer", "minimum": 0},
                        "exercise": {"type": "boolean", "description": "Hard exercise"},
                        "travel": {"type": "boolean"},
                        "sleep_loss": {"type": "boolean"},
                    },
                    "anyOf": [{"required": [field]} for field in ("label", "confounders", *storage.CONFOUNDERS)],
                },
            },
            {
                "name": "symptom_radar_labels",
                "description": "Return all logged self-report labels as [[date, label, note], ...].",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]}, None

    if method == "tools/call":
        if not isinstance(params, dict):
            raise RpcError(-32602, "params must be an object")
        name = params.get("name")
        if not isinstance(name, str):
            raise RpcError(-32602, "tool name must be a string")
        args = params.get("arguments", {})
        if not isinstance(args, dict):
            raise RpcError(-32602, "arguments must be an object")
        if name == "symptom_radar_report":
            return {"content": [{"type": "text", "text": report.build_report()}]}, None
        if name == "symptom_radar_history":
            days = args.get("days", 14)
            if type(days) is not int or not 1 <= days <= 90:
                raise RpcError(-32602, "days must be an integer between 1 and 90")
            if "metrics" in args and (not isinstance(args["metrics"], list)
                    or not all(isinstance(item, str) for item in args["metrics"])):
                raise RpcError(-32602, "metrics must be an array of strings")
            metrics = args.get("metrics") or [
                "date", "sleep_score", "night_rhr", "sleep_rhr",
                "avg_sleep_hrv", "temp_deviation",
            ]
            # Allow only known columns to prevent SQL injection.
            valid_cols = {
                "date", "sleep_score", "total_sleep_min", "sleep_efficiency",
                "deep_sleep_min", "light_sleep_min", "rem_sleep_min",
                "temp_deviation", "avg_body_temp", "night_rhr", "sleep_rhr",
                "avg_sleep_hrv", "recovery_index", "movement_index",
                "active_minutes", "inactive_time", "total_steps", "vo2_max",
                "spo2", "tosses_and_turns", "full_sleep_cycles", "hr_drop",
                "restorative_sleep", "morning_alertness", "created_at",
            }
            safe = [m for m in metrics if m in valid_cols] or ["date", "sleep_score"]
            conn = storage.init_db()
            cur = conn.execute(
                f"SELECT {', '.join(safe)} FROM daily_snapshots ORDER BY date DESC LIMIT ?",
                (days,),
            )
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            rows.reverse()
            conn.close()
            return {"content": [{"type": "text", "text": json.dumps(rows, indent=2)}]}, None
        if name == "symptom_radar_strain":
            conn = storage.init_db()
            today = config.wearer_now().strftime("%Y-%m-%d")
            history = [day for day in storage.get_recent(conn, 30) if day["date"] <= today]
            conn.close()
            if not history or history[-1]["date"] != today:
                history.append({"date": today})
            result = engine.assess_strain_result(history)
            result["level_label"] = report.STRAIN_ICONS.get(result["level"])
            return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}, None
        if name == "symptom_radar_label":
            date_str = str(args.get("date") or
                          config.wearer_now().strftime("%Y-%m-%d"))
            label = args.get("label")
            note = args.get("note")
            try:
                unknown = args.keys() - {"label", "date", "note", "confounders", *storage.CONFOUNDERS}
                if unknown:
                    raise ValueError(f"Unknown label or confounder fields: {sorted(unknown)}")
                fields = storage.parse_confounders(args.get("confounders"))
                direct = storage.parse_confounders({key: args[key] for key in storage.CONFOUNDERS if key in args})
                if fields.keys() & direct.keys():
                    raise ValueError("Duplicate confounder fields")
                fields.update(direct)
                storage.log_label(date_str, label, note, confounders=fields)
            except (ValueError, TypeError) as exc:
                return None, str(exc)
            return {"content": [{"type": "text", "text": json.dumps({
                "logged": date_str, "label": label, "note": note, "confounders": fields,
            })}]}, None
        if name == "symptom_radar_labels":
            return {"content": [{"type": "text",
                                 "text": json.dumps(storage.get_labels(), indent=2)}]}, None
        raise RpcError(-32601, f"Unknown tool: {name}")

    raise RpcError(-32601, f"Unknown method: {method}")

def _respond(req_id, *, result=None, error=None):
    response = {"jsonrpc": "2.0", "id": req_id}
    if error is not None:
        response["error"] = {"code": error.code, "message": str(error)}
    else:
        response["result"] = result
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()


def run_mcp_server():
    """Serve line-delimited JSON-RPC until EOF or shutdown, isolating each request."""
    for line in sys.stdin:
        if not line.strip():
            continue
        req_id = None
        try:
            req = json.loads(line)
        except (ValueError, RecursionError):
            _respond(None, error=RpcError(-32700, "Parse error"))
            continue
        if not isinstance(req, dict):
            _respond(None, error=RpcError(-32600, "Invalid request"))
            continue
        req_id = req.get("id")
        valid_id = req_id is None or type(req_id) in (str, int, float)
        if (req.get("jsonrpc") != "2.0" or not isinstance(req.get("method"), str)
                or not valid_id):
            _respond(req_id if valid_id else None, error=RpcError(-32600, "Invalid request"))
            continue
        method = req["method"]
        if "id" not in req:
            if method == "shutdown":
                break
            continue
        try:
            params = req.get("params", {})
            if not isinstance(params, dict):
                raise RpcError(-32602, "params must be an object")
            if method in ("ping", "shutdown"):
                result = {}
            else:
                result, error = _mcp_handle(method, params)
                if error is not None:
                    raise RpcError(-32602, error)
            _respond(req_id, result=result)
        except RpcError as exc:
            _respond(req_id, error=exc)
        except Exception:
            _respond(req_id, error=RpcError(-32603, "Internal error"))
        else:
            if method == "shutdown":
                break
