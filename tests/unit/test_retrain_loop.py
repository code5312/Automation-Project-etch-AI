"""W6-04: 자동 반복 루프(ADR-0002) 오케스트레이션 단위 테스트. fine_tune_fn/gate_fn/
regenerate_fn은 전부 스텁 — 실제 미세조정([GPU 팀원])·ViennaPS 호출은 여기서 안 한다."""

from etchpipe.retrain.loop import run_retrain_loop


def test_passes_on_first_attempt_stops_immediately() -> None:
    calls = []

    def fine_tune_fn(ckpt, cfg):
        calls.append(cfg["attempt"])
        return f"ckpt_attempt{cfg['attempt']}"

    def gate_fn(ckpt):
        return {"passed": True, "exit_code": 0}

    result = run_retrain_loop("champion.pt", fine_tune_fn, gate_fn)

    assert result["promoted"] is True
    assert result["promoted_ckpt"] == "ckpt_attempt1"
    assert len(result["attempts"]) == 1
    assert calls == [1]


def test_escalates_through_attempts_until_pass() -> None:
    def fine_tune_fn(ckpt, cfg):
        return f"ckpt_attempt{cfg['attempt']}"

    def gate_fn(ckpt):
        return {"passed": ckpt == "ckpt_attempt3", "exit_code": 0 if ckpt == "ckpt_attempt3" else 1}

    result = run_retrain_loop("champion.pt", fine_tune_fn, gate_fn)

    assert result["promoted"] is True
    assert result["promoted_ckpt"] == "ckpt_attempt3"
    assert [a["attempt"] for a in result["attempts"]] == [1, 2, 3]


def test_all_four_attempts_block_keeps_champion() -> None:
    def fine_tune_fn(ckpt, cfg):
        return f"ckpt_attempt{cfg['attempt']}"

    def gate_fn(ckpt):
        return {"passed": False, "exit_code": 1}

    result = run_retrain_loop("champion.pt", fine_tune_fn, gate_fn)

    assert result["promoted"] is False
    assert result["promoted_ckpt"] is None
    assert len(result["attempts"]) == 4
    assert [a["attempt"] for a in result["attempts"]] == [1, 2, 3, 4]


def test_regenerate_fn_called_only_before_attempt_2() -> None:
    regenerate_calls = []

    def regenerate_fn(n_train):
        regenerate_calls.append(n_train)

    def fine_tune_fn(ckpt, cfg):
        return f"ckpt_attempt{cfg['attempt']}"

    def gate_fn(ckpt):
        return {"passed": False, "exit_code": 1}

    run_retrain_loop("champion.pt", fine_tune_fn, gate_fn, regenerate_fn=regenerate_fn)

    assert regenerate_calls == [100]


def test_attempt_configs_match_adr_0002_escalation_table() -> None:
    def fine_tune_fn(ckpt, cfg):
        return f"ckpt_attempt{cfg['attempt']}"

    def gate_fn(ckpt):
        return {"passed": False, "exit_code": 1}

    result = run_retrain_loop("champion.pt", fine_tune_fn, gate_fn)
    configs = [a["config"] for a in result["attempts"]]

    assert configs[0] == {
        "attempt": 1, "n_train": 40, "epochs": (3, 5), "lr_mult": 1.0, "regenerate": False,
    }
    assert configs[1]["n_train"] == 100
    assert configs[1]["regenerate"] is True
    assert configs[2]["epochs"] == (10, 15)
    assert configs[3]["lr_mult"] == 0.5


def test_max_attempts_limits_how_far_the_loop_escalates() -> None:
    def fine_tune_fn(ckpt, cfg):
        return f"ckpt_attempt{cfg['attempt']}"

    def gate_fn(ckpt):
        return {"passed": False, "exit_code": 1}

    result = run_retrain_loop("champion.pt", fine_tune_fn, gate_fn, max_attempts=2)

    assert len(result["attempts"]) == 2
    assert result["promoted"] is False
