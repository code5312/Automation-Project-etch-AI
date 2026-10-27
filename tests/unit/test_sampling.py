"""W2-02: sampling.make_recipes/make_splits 단위 테스트."""

from etchpipe.sim.sampling import SPACE, make_ood_recipes, make_recipes, make_splits


def test_same_seed_same_result() -> None:
    a = make_recipes(8, seed=20261027)
    b = make_recipes(8, seed=20261027)
    assert a == b


def test_different_seed_different_result() -> None:
    a = make_recipes(8, seed=1)
    b = make_recipes(8, seed=2)
    assert a != b


def test_recipes_within_space_bounds() -> None:
    recipes = make_recipes(16, seed=0)
    for r in recipes:
        for name, lo, hi, _ in SPACE:
            assert lo <= r["params"][name] <= hi, name


def test_splits_have_no_overlap() -> None:
    splits = make_splits(n_train=8, n_val=4, n_test=4, n_ood=4, seed=0)
    ids = {k: {r["run_id"] for r in v} for k, v in splits.items()}
    all_ids = ids["train"] | ids["val"] | ids["test"] | ids["ood"]
    assert len(all_ids) == 8 + 4 + 4 + 4


def test_splits_sizes() -> None:
    splits = make_splits(n_train=8, n_val=4, n_test=4, n_ood=4, seed=0)
    assert len(splits["train"]) == 8
    assert len(splits["val"]) == 4
    assert len(splits["test"]) == 4
    assert len(splits["ood"]) == 4


def test_splits_same_seed_same_result() -> None:
    a = make_splits(n_train=8, n_val=4, n_test=4, n_ood=4, seed=42)
    b = make_splits(n_train=8, n_val=4, n_test=4, n_ood=4, seed=42)
    assert a == b


def test_ood_recipes_are_outside_space_bounds() -> None:
    ood = make_ood_recipes(10, seed=0)
    for r in ood:
        outside = [not (lo <= r["params"][name] <= hi) for name, lo, hi, _ in SPACE]
        assert any(outside), r
