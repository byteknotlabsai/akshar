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


def _feeder(tmp_path, shares):
    config = {"pool": str(tmp_path / "pool"), "pool_gb": 1, "floor_gb": 0, "rclone": ["rclone"], "sources": [
        {"name": name, "remote": f"r:{name}", "share": share, "manifest": str(tmp_path / f"{name}.jsonl"),
         "labels": str(tmp_path / f"{name}.labels.jsonl")} for name, share in shares.items()]}
    (tmp_path / "feed.json").write_text(json.dumps(config))
    feeder = feed.Feeder(tmp_path / "feed.json")
    for source in feeder.sources.values():  # one shard each, as if listed on Drive and labelled
        source.by_shard, source.sizes = {"s0": [("{}", "{}", 3600.0)]}, {"s0": 1}
    return feeder


def test_pick_ranks_sources_by_relative_shortfall_with_absent_sources_first(tmp_path):
    # Planned shares: nptel 10%, indicvoices 22%, kathbath 5%. In the pool: an nptel shard of 30 h and 20 h of
    # indicvoices. Kathbath holds nothing, so it comes first although its shortfall in hours is the smallest (D31).
    feeder = _feeder(tmp_path, {"nptel": 0.10, "indicvoices": 0.22, "kathbath": 0.05})
    feeder.hours = {"nptel__a": 30.0, "indicvoices__b": 20.0}
    assert feeder.pick()[0].name == "kathbath"
    # With 4 h of Kathbath in (3.5% against 5%: 70% of its share), IndicVoices at 20 of 54 h (37% against 22%) and
    # NPTEL at 30 of 54 h (56% against 10%) are both over, so Kathbath is still the shortest relative to its share.
    feeder.hours["kathbath__c"] = 4.0
    assert feeder.pick()[0].name == "kathbath"
    feeder.hours["kathbath__c"] = 10.0  # 17% against 5%: now IndicVoices (37% / 22%) is the shortest
    assert feeder.pick()[0].name == "indicvoices"


def test_relist_writes_the_shares_for_the_run(tmp_path, monkeypatch):
    feeder = _feeder(tmp_path, {"a": 0.7, "b": 0.3, "off": 0.0})
    monkeypatch.setattr(feed.Source, "load", lambda self: None)
    monkeypatch.setattr(feed.Source, "list_tars", lambda self: None)
    feeder.relist()
    assert json.loads((tmp_path / "pool" / "shares.json").read_text())["shares"] == {"a": 0.7, "b": 0.3}
