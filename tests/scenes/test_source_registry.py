from qdgrasp.scenes.source_registry import SOURCE_IDS, load_source_records


def test_external_source_records_pin_license_layout_and_fail_closed_policy():
    records = load_source_records()
    assert tuple(records) == SOURCE_IDS
    for dataset_id, record in records.items():
        assert record["dataset_id"] == dataset_id
        assert record["license"]
        assert record["source_url"].startswith("https://")
        assert record["evidence_url"].startswith("https://")
        assert record["expected_layout"]
        assert record["redistributable"] is False
