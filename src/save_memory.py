import json
from datetime import datetime
from pathlib import Path

# Define file paths
chatbot_memory_file = Path("./data/logs/chatbot_memory.json")
workshop_memory_file = Path("./data/logs/workshop_memory.json")

# ─── Chatbot Memory Helpers ────────────────────────────────────────

def save_chatbot_memory(input_text: str, output_text: str):
    entry = {"input": input_text, "output": output_text}
    with open(chatbot_memory_file, "a") as f:
        json.dump(entry, f)
        f.write("\n")

DEFAULT_ROLES        = ["facilitator", "hr", "strategy"]

def load_chatbot_memory():
    """
    Load chatbot memory as a list of {"timestamp", "input", "output"} entries.
    If the file is missing or contains invalid JSON, returns an empty list.
    """
    try:
        with open(chatbot_memory_file, "r") as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        # File doesn’t exist yet
        return []
    except json.JSONDecodeError:
        # File exists but has malformed/empty JSON
        return []

def load_workshop_memory():
    """
    Load workshop memory as a dict of timestamped lists under keys 'facilitator', 'hr', 'strategy'.
    If the file is missing or contains invalid JSON, returns an initialized dict.
    """
    try:
        text = workshop_memory_file.read_text()
        data = json.loads(text) if text.strip() else {}
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    # Ensure each expected role key exists and is a list
    for role in DEFAULT_ROLES:
        data.setdefault(role, [])

    return data

def save_workshop_memory(role: str, message: str, timestamp: str = None):
    """
    Append a timestamped entry to the list under `role` in workshop_memory.json.
    role must be 'facilitator', 'hr', or 'strategy'.
    """
    data = load_workshop_memory()
    role_key = role.lower()
    if role_key not in data:
        raise ValueError(f"Unknown role: {role!r}")
    # Use provided timestamp or current time
    ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M")
    
    data[role_key].append({"timestamp": ts, "message": message})
    # Sort each list by timestamp to ensure chronological order
    for lst in data.values():
        lst.sort(key=lambda e: datetime.strptime(e["timestamp"], "%Y-%m-%d %H:%M"))
    # Write back entire dict
    workshop_memory_file.write_text(json.dumps(data, indent=2))

def load_workshop_history_for_agent():
    """Return the workshop memory dict as is (timestamped entries)."""
    return load_workshop_memory()

def get_combined_history():
    return {
        "chatbot_history": load_chatbot_memory(),
        "workshop_history": load_workshop_history_for_agent()
    }
