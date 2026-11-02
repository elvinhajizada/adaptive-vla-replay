import pytest

from adaptive_vla import subgoals as sg

GOALS = ["Close drawer_bottom", "Open drawer_top"]


def _rec(init, final, ever=None, success=None, at_reset=(False, False), task=23):
    return {"task_id": task, "init_state": init, "goals": GOALS, "at_reset": list(at_reset),
            "ever": list(ever or final), "final": list(final),
            "success": all(final) if success is None else success, "steps": 10}


def test_summarize_rates_and_partial():
    recs = [_rec(0, [True, True]), _rec(1, [True, False]), _rec(2, [False, False], ever=[True, False]),
            _rec(3, [True, False], at_reset=(True, False))]
    s = sg.summarize(recs)[23]
    assert (s["n"], s["success"]) == (4, 1)
    assert s["partial"] == 1            # init 1 only: init 3's drawer was already closed at reset
    assert s["goals"][0]["final"] == 0.75 and s["goals"][0]["ever"] == 1.0
    assert s["goals"][1]["final"] == 0.25 and s["goals"][0]["at_reset"] == 0.25


def test_summarize_rejects_duplicate_init_states():
    with pytest.raises(ValueError):
        sg.summarize([_rec(0, [True, True]), _rec(0, [False, False])])


def test_check_against_eval_info():
    s = sg.summarize([_rec(0, [True, True]), _rec(1, [False, False])])
    sg.check_against_eval_info(s, {"per_task": [{"task_id": 23, "n_episodes": 2, "n_success": 1}]})
    with pytest.raises(ValueError):
        sg.check_against_eval_info(s, {"per_task": [{"task_id": 23, "n_episodes": 2, "n_success": 2}]})
