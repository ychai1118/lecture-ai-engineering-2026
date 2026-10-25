import json
from types import SimpleNamespace

from google.genai import types

import app.ask as ask_module


def test_ask_restores_and_saves_history(monkeypatch, tmp_path):
    history_path = tmp_path / "ask_history.json"
    created_histories = []

    class FakeChat:
        def __init__(self, history):
            self.history = list(history)

        def send_message(self, message):
            self.history.extend(
                [
                    types.Content(role="user", parts=[types.Part(text=message)]),
                    types.Content(role="model", parts=[types.Part(text="reply")]),
                ]
            )
            return SimpleNamespace(usage_metadata=None, text="reply")

        def get_history(self):
            return self.history

    class FakeChats:
        def create(self, *, model, history):
            created_histories.append(list(history))
            return FakeChat(history)

    class FakeClient:
        def __init__(self, **kwargs):
            self.chats = FakeChats()

    monkeypatch.setattr(ask_module, "HISTORY_PATH", history_path)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask_module.genai, "Client", FakeClient)

    assert ask_module.ask("first") == "reply"
    assert ask_module.ask("follow-up") == "reply"

    assert created_histories[0] == []
    assert len(created_histories[1]) == 2
    assert created_histories[1][0].parts[0].text == "first"
    assert created_histories[1][1].parts[0].text == "reply"
    assert len(json.loads(history_path.read_text())[ask_module.MODEL]) == 4
