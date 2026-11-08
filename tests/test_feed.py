import json

from akshar.train import feed


def test_rows_of_skips_a_half_written_last_line(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text('{"id": "a"}\n{"id": "b"}\n{"id": "c", "aud')
    assert [row["id"] for _, row in feed.rows_of(path)] == ["a", "b"]


def test_a_source_offers_labelled_shards_on_drive_least_fetched_first(tmp_path):
    rows = [{"id": f"s/{i}", "audio": f"clips/shard-{i % 3}/{i}.flac", "duration": 5.0} for i in range(9)]
    (tmp_path / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "labels.jsonl").write_text("".join(json.dumps({"id": r["id"], "teacher": "x"}) + "\n"
                                                   for r in rows if r["id"] != "s/2"))
    (tmp_path / "exclude.jsonl").write_text(json.dumps({"id": "s/0"}) + "\n")
    source = feed.Source({"name": "s", "remote": "r:x", "share": 1.0, "manifest": str(tmp_path / "manifest.jsonl"),
                          "labels": str(tmp_path / "labels.jsonl"), "exclude": [str(tmp_path / "exclude.jsonl")]}, [])
    source.load()
    assert {k: len(v) for k, v in source.by_shard.items()} == {"shard-0": 2, "shard-1": 3, "shard-2": 2}
    source.sizes = {"shard-0": 1, "shard-1": 1}  # shard-2 is not on Drive yet
    source.visits = {"shard-0": 1}
    import random
    assert source.next_shard(set(), random.Random(0)) == "shard-1"
    assert source.next_shard({"s__shard-1"}, random.Random(0)) == "shard-0"
