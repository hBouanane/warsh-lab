"""Splitting: coverage, determinism, small groups, and leakage detection."""

import pytest

from warshlab import splits as S


def corpus(per_group=10, groups=("alpha", "beta", "gamma")):
    return [
        {"segment_id": f"{group}-{i:03d}", "reciter_slug": group}
        for group in groups
        for i in range(per_group)
    ]


# --------------------------------------------------------------------------
# Stratified
# --------------------------------------------------------------------------


def test_stratified_split_keeps_every_item():
    items = corpus()
    result = S.stratified_split(items)
    assert sum(result.sizes.values()) == len(items)


def test_stratified_split_puts_every_group_in_every_split():
    result = S.stratified_split(corpus())
    for name in ("train", "val", "test"):
        assert set(result.groups_per_split[name]) == {"alpha", "beta", "gamma"}


def test_stratified_split_respects_the_ratios():
    result = S.stratified_split(corpus(per_group=100), ratios=(0.8, 0.1, 0.1))
    assert result.sizes["train"] == 240
    assert result.sizes["val"] == 30
    assert result.sizes["test"] == 30


def test_stratified_split_is_deterministic():
    first = S.stratified_split(corpus(), seed=7)
    second = S.stratified_split(corpus(), seed=7)
    assert first.sizes == second.sizes
    assert [i["segment_id"] for i in first["val"]] == [
        i["segment_id"] for i in second["val"]
    ]


def test_a_different_seed_gives_a_different_assignment():
    first = {i["segment_id"] for i in S.stratified_split(corpus(), seed=1)["val"]}
    second = {i["segment_id"] for i in S.stratified_split(corpus(), seed=2)["val"]}
    assert first != second


def test_adding_a_group_does_not_reshuffle_the_existing_ones():
    """Split stability under corpus growth -- otherwise every new reciter
    invalidates every previously reported validation score."""
    before = S.stratified_split(corpus(groups=("alpha", "beta")), seed=3)
    after = S.stratified_split(corpus(groups=("alpha", "beta", "delta")), seed=3)

    def alpha_val(result):
        return sorted(
            i["segment_id"] for i in result["val"] if i["reciter_slug"] == "alpha"
        )

    assert alpha_val(before) == alpha_val(after)


def test_splits_do_not_overlap():
    result = S.stratified_split(corpus())
    assert S.check_leakage(result.splits) == []


def test_output_is_interleaved_rather_than_grouped_by_reciter():
    """Training must not see one voice for a long consecutive stretch."""
    train = S.stratified_split(corpus(per_group=40))["train"]
    groups = [item["reciter_slug"] for item in train]
    switches = sum(1 for a, b in zip(groups, groups[1:]) if a != b)
    assert switches > len(groups) // 4


def test_a_group_smaller_than_the_split_count_is_reported_not_dropped():
    items = corpus(groups=("big",)) + [{"segment_id": "tiny-1", "reciter_slug": "tiny"}]
    result = S.stratified_split(items)
    assert sum(result.sizes.values()) == len(items)
    assert any("tiny" in w for w in result.warnings)
    assert "tiny" in result.groups_per_split["train"]


def test_no_split_is_left_empty_when_the_group_is_big_enough():
    result = S.stratified_split(corpus(groups=("solo",), per_group=3))
    assert all(size >= 1 for size in result.sizes.values())


def test_ratio_and_name_counts_must_match():
    with pytest.raises(ValueError, match="ratios"):
        S.stratified_split(corpus(), ratios=(0.5, 0.5), names=("a", "b", "c"))


def test_negative_ratios_are_rejected():
    with pytest.raises(ValueError):
        S.stratified_split(corpus(), ratios=(-1.0, 1.0, 1.0))


def test_missing_group_key_raises_a_clear_error():
    with pytest.raises(KeyError, match="reciter_slug"):
        S.stratified_split([{"segment_id": "x"}])


def test_two_way_split_is_supported():
    result = S.stratified_split(corpus(), ratios=(0.9, 0.1), names=("train", "val"))
    assert set(result.sizes) == {"train", "val"}


def test_summary_mentions_every_split():
    text = S.stratified_split(corpus()).summary()
    for name in ("train", "val", "test"):
        assert name in text


# --------------------------------------------------------------------------
# Holdout
# --------------------------------------------------------------------------


def test_holdout_split_removes_the_group_entirely():
    result = S.holdout_split(corpus(), holdout=["beta"])
    assert "beta" not in result.groups_per_split["seen"]
    assert result.groups_per_split["unseen"] == ["beta"]


def test_holdout_split_keeps_every_item():
    items = corpus()
    result = S.holdout_split(items, holdout=["beta"])
    assert sum(result.sizes.values()) == len(items)


def test_holdout_split_picks_groups_when_none_are_named():
    result = S.holdout_split(corpus(groups=tuple(f"r{i}" for i in range(6))), n_holdout=2)
    assert len(result.groups_per_split["unseen"]) == 2


def test_holdout_split_is_deterministic():
    groups = tuple(f"r{i}" for i in range(6))
    first = S.holdout_split(corpus(groups=groups), n_holdout=2, seed=5)
    second = S.holdout_split(corpus(groups=groups), n_holdout=2, seed=5)
    assert first.groups_per_split["unseen"] == second.groups_per_split["unseen"]


def test_holdout_split_refuses_to_hold_out_everything():
    with pytest.raises(ValueError, match="nothing would be left"):
        S.holdout_split(corpus(), n_holdout=3)


def test_unknown_holdout_group_is_warned_about():
    result = S.holdout_split(corpus(), holdout=["nobody"])
    assert any("nobody" in w for w in result.warnings)


# --------------------------------------------------------------------------
# Leakage
# --------------------------------------------------------------------------


def test_check_leakage_finds_an_id_in_two_splits():
    shared = {"segment_id": "dup", "reciter_slug": "x"}
    problems = S.check_leakage({"train": [shared], "val": [shared]})
    assert len(problems) == 1
    assert "dup" in problems[0]


def test_check_leakage_finds_a_duplicate_within_one_split():
    shared = {"segment_id": "dup", "reciter_slug": "x"}
    problems = S.check_leakage({"train": [shared, shared]})
    assert any("2 times" in p for p in problems)


def test_check_leakage_is_quiet_when_splits_are_clean():
    assert S.check_leakage({"train": corpus(groups=("a",)), "val": corpus(groups=("b",))}) == []


def test_group_summary_orders_by_size():
    items = corpus(per_group=5, groups=("small",)) + corpus(per_group=20, groups=("big",))
    assert S.group_summary(items)[0] == ("big", 20)
