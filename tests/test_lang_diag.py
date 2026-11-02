"""CPU tests for the language diagnostic's pure parts (adaptive_vla/lang_diag.py). Stdlib only."""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_vla import lang_diag as ld  # noqa: E402

LANG = [f"instruction {k}" for k in range(10)]


def test_parse_tasks():
    assert ld.parse_tasks("8,9,7") == [8, 9, 7]
    assert ld.parse_tasks("[8, 9, 7]") == [8, 9, 7]
    with pytest.raises(ValueError):
        ld.parse_tasks("8,8")
    with pytest.raises(ValueError):
        ld.parse_tasks("")


def test_cyclic_pairs_is_a_derangement_using_each_task_once():
    tasks = [8, 9, 7]
    pairs = ld.cyclic_pairs(tasks)
    assert pairs == {8: 9, 9: 7, 7: 8}
    assert all(a != b for a, b in pairs.items())
    assert sorted(pairs.values()) == sorted(tasks)
    assert ld.cyclic_pairs([3, 5]) == {3: 5, 5: 3}
    with pytest.raises(ValueError):
        ld.cyclic_pairs([3])


def test_parse_and_validate_pairs():
    assert ld.parse_pairs("8:9, 9:7") == {8: 9, 9: 7}
    with pytest.raises(ValueError):
        ld.parse_pairs("8:9,8:7")                       # A paired twice
    ld.validate_pairs([8], {8: 9}, 10)                 # B need not be among the A tasks
    with pytest.raises(ValueError):
        ld.validate_pairs([8, 9], {8: 9}, 10)          # 9 has no B
    with pytest.raises(ValueError):
        ld.validate_pairs([8], {8: 8}, 10)             # self pair
    with pytest.raises(ValueError):
        ld.validate_pairs([8], {8: 10}, 10)            # B out of range
    with pytest.raises(ValueError):
        ld.validate_pairs([12], {12: 1}, 10)           # A out of range


def test_build_specs_conditions():
    specs = ld.build_specs([8, 9], {8: 9, 9: 8}, LANG)
    assert [(s.condition, s.task_id) for s in specs] == [
        ("true", 8), ("true", 9), ("blank", 8), ("blank", 9), ("swapped", 8), ("swapped", 9)]
    true8, blank8, swap8 = specs[0], specs[2], specs[4]
    assert (true8.instruction, true8.instructed_task_id, true8.scored_task_id) == ("instruction 8", 8, 8)
    assert (blank8.instruction, blank8.instructed_task_id, blank8.scored_task_id) == ("", None, 8)
    # swapped: A's env and init state, B's instruction, scored by B's goal
    assert (swap8.task_id, swap8.instruction, swap8.scored_task_id) == (8, "instruction 9", 9)
    assert all(s.partner_task_id == {8: 9, 9: 8}[s.task_id] for s in specs)


def test_build_specs_subset_and_blank_text():
    specs = ld.build_specs([8], {8: 9}, LANG, conditions=["swapped", "blank"], blank="do the task")
    assert [s.condition for s in specs] == ["blank", "swapped"]       # canonical order
    assert specs[0].instruction == "do the task"
    with pytest.raises(ValueError):
        ld.build_specs([8], {8: 9}, LANG, conditions=["shuffled"])


BDDL = """(define (problem P)
  (:domain robosuite)
  (:language {lang})
  (:regions (r (:target t)))
  (:objects bowl_1 - bowl)
  (:obj_of_interest
    {ooi}
  )
  (:init (On bowl_1 r))
  (:goal
    (And ({goal}))
  )
)"""


def test_bddl_scene_and_goal_text():
    a = BDDL.format(lang="Put it here", ooi="bowl_1", goal="On bowl_1 r")
    b = BDDL.format(lang="Lift it", ooi="", goal="Lifted bowl_1")
    assert ld.bddl_scene_text(a) == ld.bddl_scene_text(b)
    assert ld.bddl_goal_text(a) == "(:goal (And (On bowl_1 r)) )"
    ld.check_shared_scene({"a": a, "b": b})
    with pytest.raises(ValueError):                                   # same goal twice
        ld.check_shared_scene({"a": a, "a2": a})
    c = a.replace("bowl_1 - bowl", "bowl_1 plate_1 - bowl")
    with pytest.raises(ValueError):                                   # different scene
        ld.check_shared_scene({"a": a, "c": c})


def _libero_goal_dir():
    spec = importlib.util.find_spec("libero")       # locates the package without importing it
    if spec is None or not spec.submodule_search_locations:
        return None
    d = Path(list(spec.submodule_search_locations)[0]) / "libero" / "bddl_files" / "libero_goal"
    return d if d.is_dir() else None


@pytest.mark.skipif(_libero_goal_dir() is None, reason="hf-libero not installed")
def test_installed_libero_goal_tasks_share_one_scene():
    texts = {p.name: p.read_text() for p in sorted(_libero_goal_dir().glob("*.bddl"))}
    assert len(texts) == 10
    ld.check_shared_scene(texts)


def test_mask_bits():
    assert ld.mask_bits(0.0) == []
    assert ld.mask_bits(float(1 << 9 | 1 << 7)) == [7, 9]
    with pytest.raises(ValueError):
        ld.mask_bits(0.5)


def _eval_info(cells):
    """cells: {(condition, A): [(success, goals reached), ...]} -> eval_info.json shape."""
    return {"per_task": [
        {"task_group": c, "task_id": a, "metrics": {
            "successes": [s for s, _ in eps],
            "max_rewards": [float(sum(1 << g for g in goals)) for _, goals in eps]}}
        for (c, a), eps in cells.items()]}


def test_summarize_counts():
    specs = ld.build_specs([8, 9], {8: 9, 9: 8}, LANG)
    info = _eval_info({
        ("true", 8): [(True, [8]), (False, [])],
        ("true", 9): [(True, [9]), (True, [9, 8])],
        ("blank", 8): [(False, [9]), (True, [8])],
        ("blank", 9): [(False, []), (False, [3])],
        ("swapped", 8): [(False, [8]), (True, [9])],        # ep 0: did A (8) instead of B (9)
        ("swapped", 9): [(False, [9]), (False, [])],        # ep 0: did A (9) instead of B (8)
    })
    s = ld.summarize(info, specs)
    c = s["conditions"]
    assert c["true"]["success"]["k"] == 3 and c["true"]["success"]["n"] == 4
    assert c["true"]["partner_goal"]["k"] == 1
    assert c["blank"]["own_goal"]["k"] == 1 and c["blank"]["partner_goal"]["k"] == 1
    assert c["blank"]["no_goal"]["k"] == 1
    assert c["swapped"]["success"]["k"] == 1
    assert c["swapped"]["own_goal"]["k"] == 2
    assert c["swapped"]["own_not_partner"]["k"] == 2
    assert c["swapped"]["per_task"][8] == {"success": 1, "own_goal": 1, "partner_goal": 1, "n": 2}
    assert len(s["episodes"]) == 12
    text = ld.format_summary(s)
    assert "swapped" in text and "A=8 B=9" in text


def test_summarize_rejects_success_that_is_not_the_scored_goal():
    specs = ld.build_specs([8], {8: 9}, LANG, conditions=["swapped"])
    # success flagged, but only A's goal (8) was reached: the env scored the wrong goal
    with pytest.raises(ValueError):
        ld.summarize(_eval_info({("swapped", 8): [(True, [8])]}), specs)


def test_wilson_ci95():
    lo, hi = ld.wilson_ci95(64, 100)
    assert (lo, hi) == (54.2, 72.7)
    assert ld.wilson_ci95(0, 10)[0] == 0.0
