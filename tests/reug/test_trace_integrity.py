from reug.events import EventEmitter


def test_emit_event(tmp_path):
    path=tmp_path/"events.jsonl"
    ee=EventEmitter(lambda e: open(path,"a").write(str(e)+"\n"))
    ee.emit({"event_type":"STATE_TRANSITION","from":"A","to":"B"})
    txt=path.read_text()
    assert "STATE_TRANSITION" in txt

