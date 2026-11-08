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


def test_a_share_of_english_clips_is_tagged_hindi(tmp_path):
    rows = [{"id": f"cv/{i}", "audio": f"clips/cv-0/{i}.flac", "duration": 4.0} for i in range(400)]
    (tmp_path / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "labels.jsonl").write_text("".join(json.dumps({"id": r["id"], "teacher": "hello there", "language": "en"})
                                                   + "\n" for r in rows))
    spec = {"name": "cv", "remote": "r:x", "share": 1.0, "manifest": str(tmp_path / "manifest.jsonl"),
            "labels": str(tmp_path / "labels.jsonl"), "hi_tag_share": 0.2}
    source = feed.Source(spec, [])
    source.load()
    labels = [json.loads(label) for _, label, _ in source.by_shard["cv-0"]]
    hindi = [x for x in labels if x["language"] == "hi"]
    assert 50 < len(hindi) < 110 and all(x["teacher"] == "hello there" for x in hindi)
    source.load()  # unchanged files and share: the same clips stay tagged
    assert [json.loads(label)["language"] for _, label, _ in source.by_shard["cv-0"]] == [x["language"] for x in labels]
