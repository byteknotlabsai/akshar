from akshar.files import repair_jsonl


def test_repair_cuts_only_a_partial_last_line(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_bytes(b'{"a": 1}\n{"b": 2}\n{"c": ')
    assert repair_jsonl(path) == len(b'{"c": ')
    assert path.read_bytes() == b'{"a": 1}\n{"b": 2}\n'
    assert repair_jsonl(path) == 0
    path.write_bytes(b'{"half')
    assert repair_jsonl(path) == 6 and path.read_bytes() == b""
