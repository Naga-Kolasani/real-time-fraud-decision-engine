"""
Quick checks on the decision logic and row-building helper.
No model, no data, no server - just the pure functions.
"""

import pytest

from src.models.infer import T1, T2, _build_feature_row, decide


def test_decide_approve():
    assert decide(0.0) == "approve"


def test_decide_review():
    midpoint = (T1 + T2) / 2
    assert decide(midpoint) == "review"


def test_decide_block():
    assert decide(1.0) == "block"


def test_decide_boundary_t1():
    # T1 itself belongs to review, not approve.
    assert decide(T1) == "review"


def test_decide_boundary_t2():
    #T2 itself belongs to block, not review.
    assert decide(T2) == "block"


def test_build_feature_row_preserves_column_order():
    #Dict order deliberately doesn't match feature_columns order.
    transaction = {"Amount": 100.0, "Time": 5.0, "V1": 0.1}
    feature_columns = ["Time", "Amount", "V1"]

    row = _build_feature_row(transaction, feature_columns)

    assert list(row.columns) == feature_columns


def test_build_feature_row_one_row():
    transaction = {"Time": 5.0, "Amount": 100.0}
    feature_columns = ["Time", "Amount"]

    row = _build_feature_row(transaction, feature_columns)

    assert len(row) == 1


def test_build_feature_row_missing_feature_raises():
    # Keeping this simple - just check the error mentions what's missing.
    transaction = {"Time": 5.0}
    feature_columns = ["Time", "Amount"]

    with pytest.raises(ValueError, match="missing required feature"):
        _build_feature_row(transaction, feature_columns)
