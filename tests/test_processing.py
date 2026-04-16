import pandas as pd
import pytest

import healthcare_nlp.data_preprocessing as m


def _df(
    text=("a", "b"),
    label=(0, 1),
    label_text=("neg", "pos"),
):
    return pd.DataFrame(
        {
            "text": list(text),
            "label": list(label),
            "label_text": list(label_text),
        }
    )


def test_validate_columns_raises_on_missing_columns():
    df = pd.DataFrame({"text": ["a"], "label": [0]})
    with pytest.raises(ValueError, match="Missing required columns"):
        m._validate_columns(df)


def test_validate_labels_raises_on_unexpected_values():
    df = _df(label=(0, 2))
    with pytest.raises(ValueError, match="Unexpected label values"):
        m._validate_labels(df)


def test_add_normalized_text_adds_column_and_keeps_input_unchanged(monkeypatch):
    df = _df(text=(" A ", "B!"))
    df_before = df.copy(deep=True)

    monkeypatch.setattr(m, "normalize_text", lambda x: x.strip().lower())

    out = m.add_normalized_text(df)

    assert "_norm_text" in out.columns
    assert out["_norm_text"].tolist() == ["a", "b!"]
    pd.testing.assert_frame_equal(df, df_before)


def test_add_normalized_text_raises_when_requested_column_is_missing(monkeypatch):
    df = _df()
    monkeypatch.setattr(m, "normalize_text", lambda x: x)

    with pytest.raises(KeyError):
        m.add_normalized_text(df, col="other")


def test_drop_label_conflicts_removes_only_conflicting_groups():
    df = pd.DataFrame(
        {
            "text": ["a", "a2", "b", "b2", "c"],
            "label": [0, 1, 0, 0, 1],
            "label_text": ["x", "y", "x", "x", "y"],
            "_norm_text": ["same", "same", "keep0", "keep0", "keep1"],
        }
    )

    out = m.drop_label_conflicts(df)

    assert out["_norm_text"].tolist() == ["keep0", "keep0", "keep1"]
    assert out["text"].tolist() == ["b", "b2", "c"]


def test_drop_label_conflicts_raises_without_norm_text():
    df = _df()

    with pytest.raises(ValueError, match="Missing required column"):
        m.drop_label_conflicts(df)


def test_dedupe_within_split_keeps_first_occurrence_per_normalized_text():
    df = pd.DataFrame(
        {
            "text": ["a", "a2", "b"],
            "label": [0, 0, 1],
            "label_text": ["x", "x", "y"],
            "_norm_text": ["same", "same", "other"],
        }
    )

    out = m.dedupe_within_split(df)

    assert out["_norm_text"].tolist() == ["same", "other"]
    assert out["text"].tolist() == ["a", "b"]


def test_dedupe_within_split_raises_without_norm_text():
    df = _df()

    with pytest.raises(ValueError, match="Missing required column"):
        m.dedupe_within_split(df)


def test_remove_train_test_overlap_removes_overlap_from_train_only_and_preserves_inputs():
    df_train = pd.DataFrame(
        {
            "text": ["a", "b", "c"],
            "label": [0, 1, 0],
            "label_text": ["x", "y", "x"],
            "_norm_text": ["same", "keep1", "keep2"],
        }
    )
    df_test = pd.DataFrame(
        {
            "text": ["z"],
            "label": [1],
            "label_text": ["y"],
            "_norm_text": ["same"],
        }
    )
    df_train_before = df_train.copy(deep=True)
    df_test_before = df_test.copy(deep=True)

    train_out, test_out = m.remove_train_test_overlap(df_train, df_test)

    assert train_out["_norm_text"].tolist() == ["keep1", "keep2"]
    pd.testing.assert_frame_equal(test_out, df_test)
    pd.testing.assert_frame_equal(df_train, df_train_before)
    pd.testing.assert_frame_equal(df_test, df_test_before)


def test_remove_train_test_overlap_raises_without_norm_text():
    df_train = _df()
    df_test = _df().assign(_norm_text=["a", "b"])

    with pytest.raises(ValueError, match="Missing required column"):
        m.remove_train_test_overlap(df_train, df_test)


def test_train_val_split_returns_partitioned_stratified_data():
    df = pd.DataFrame(
        {
            "text": [f"t{i}" for i in range(20)],
            "label": [0] * 10 + [1] * 10,
            "label_text": ["neg"] * 10 + ["pos"] * 10,
        }
    )

    train_df, val_df = m.train_val_split(df, val_size=0.2)

    assert len(train_df) == 16
    assert len(val_df) == 4
    assert train_df["label"].value_counts().to_dict() == {0: 8, 1: 8}
    assert val_df["label"].value_counts().to_dict() == {0: 2, 1: 2}


def test_train_val_split_raises_on_invalid_val_size():
    df = _df()

    with pytest.raises(ValueError, match="val_size must be between 0 and 1"):
        m.train_val_split(df, val_size=0)


def test_train_val_split_raises_when_stratification_is_impossible():
    df = pd.DataFrame(
        {
            "text": ["a", "b", "c"],
            "label": [0, 0, 1],
            "label_text": ["neg", "neg", "pos"],
        }
    )

    with pytest.raises(ValueError):
        m.train_val_split(df, val_size=0.5)


def test_clean_splits_runs_pipeline_end_to_end(monkeypatch):
    df_train = pd.DataFrame(
        {
            "text": [
                "A", "A ",
                "B", "B ",
                "C",
                "D", "D!",
                "F", "F!",
                "G",
                "H", "I",
            ],
            "label": [
                0, 1,
                0, 0,
                0,
                1, 1,
                1, 1,
                0,
                0, 1,
            ],
            "label_text": [
                "neg", "pos",
                "neg", "neg",
                "neg",
                "pos", "pos",
                "pos", "pos",
                "neg",
                "neg", "pos",
            ],
        }
    )
    df_test = pd.DataFrame(
        {
            "text": ["c", "E"],
            "label": [0, 1],
            "label_text": ["neg", "pos"],
        }
    )

    monkeypatch.setattr(m, "load_train_test_data", lambda: (df_train, df_test))
    monkeypatch.setattr(m, "normalize_text", lambda x: str(x).strip().lower())

    train_out, val_out, test_out = m.clean_splits(0.25)

    combined_train = pd.concat([train_out, val_out], ignore_index=True)

    assert "_norm_text" in train_out.columns
    assert "_norm_text" in val_out.columns
    assert "_norm_text" in test_out.columns
    assert set(combined_train["_norm_text"]).isdisjoint(set(test_out["_norm_text"]))
    assert "a" not in set(combined_train["_norm_text"])
    assert combined_train["_norm_text"].tolist().count("b") == 1
    assert set(combined_train["_norm_text"]) == {"b", "d", "d!", "f", "f!", "g", "h", "i"}
    assert set(test_out["_norm_text"]) == {"c", "e"}


def test_clean_splits_raises_when_cleaning_leaves_too_few_samples_per_class(monkeypatch):
    df_train = pd.DataFrame(
        {
            "text": ["A", "A ", "B", "C", "D"],
            "label": [0, 1, 0, 0, 1],
            "label_text": ["neg", "pos", "neg", "neg", "pos"],
        }
    )
    df_test = pd.DataFrame(
        {
            "text": ["c", "E"],
            "label": [0, 1],
            "label_text": ["neg", "pos"],
        }
    )

    monkeypatch.setattr(m, "load_train_test_data", lambda: (df_train, df_test))
    monkeypatch.setattr(m, "normalize_text", lambda x: str(x).strip().lower())

    with pytest.raises(ValueError):
        m.clean_splits()


def test_save_clean_splits_writes_parquet_files(tmp_path):
    df_train = _df()
    df_val = _df(text=("c",), label=(0,), label_text=("neg",))
    df_test = _df(text=("d",), label=(1,), label_text=("pos",))

    m.save_clean_splits(df_train, df_val, df_test, tmp_path)

    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "train.parquet"), df_train)
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "val.parquet"), df_val)
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "test.parquet"), df_test)
