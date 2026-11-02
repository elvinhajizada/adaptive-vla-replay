import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from adaptive_vla import checkpoints as ck


def _dirs(tmp_path):
    base, run = tmp_path / "theta0", tmp_path / "run" / "checkpoints" / "last"
    pol = run / "pretrained_model"
    base.mkdir(parents=True)
    pol.mkdir(parents=True)
    (run / "training_state").mkdir()
    (run / "training_state" / "optimizer.safetensors").write_bytes(b"x")
    w = {"vlm.w": torch.randn(4, 4).to(torch.bfloat16), "expert.w": torch.randn(3, 2).to(torch.bfloat16),
         "expert.b": torch.zeros(2, dtype=torch.bfloat16)}
    save_file(w, str(base / ck.MODEL))
    tuned = dict(w, **{"expert.w": w["expert.w"] + 1})
    save_file(tuned, str(pol / ck.MODEL))
    (pol / "config.json").write_text("{}")
    return base, pol, tuned


def test_slim_keeps_only_changed_and_rebuilds_exactly(tmp_path):
    base, pol, tuned = _dirs(tmp_path)
    info = ck.slim_checkpoint(str(pol), str(base), expected_params=6)
    assert info["n_tensors"] == 1 and info["n_params"] == 6
    assert not (pol / ck.MODEL).exists() and not (pol.parent / "training_state").exists()
    assert set(load_file(str(pol / ck.DELTA))) == {"expert.w"}
    out = ck.materialize(str(pol), str(tmp_path / "full"))
    rebuilt = load_file(str(out / ck.MODEL))
    assert all(torch.equal(rebuilt[k], tuned[k]) for k in tuned)
    assert (out / "config.json").exists() and not (out / ck.DELTA).exists()
    assert json.loads((out / "materialized_from.json").read_text())["source"] == str(pol.resolve())
    assert ck.slim_checkpoint(str(pol), str(base)) == json.loads((pol / ck.SLIM_INFO).read_text())


def test_slim_refuses_when_frozen_weights_moved(tmp_path):
    base, pol, _ = _dirs(tmp_path)
    with pytest.raises(ValueError):
        ck.slim_checkpoint(str(pol), str(base), expected_params=2)
    assert (pol / ck.MODEL).exists()


def test_changed_tensors_rejects_key_mismatch():
    with pytest.raises(ValueError):
        ck.changed_tensors({"a": torch.zeros(1)}, {"b": torch.zeros(1)})


def test_interpolate_endpoints_and_midpoint(tmp_path):
    base, pol, tuned = _dirs(tmp_path)
    ck.slim_checkpoint(str(pol), str(base))
    w0 = load_file(str(base / ck.MODEL))
    for alpha, want in ((0.0, w0["expert.w"]), (1.0, tuned["expert.w"]),
                        (0.5, (w0["expert.w"].float() + 0.5 * (tuned["expert.w"].float()
                                                               - w0["expert.w"].float())).to(torch.bfloat16))):
        out = ck.interpolate(str(pol), str(tmp_path / f"a{alpha}"), alpha)
        assert json.loads((out / ck.SLIM_INFO).read_text())["alpha"] == alpha
        full = load_file(str(ck.materialize(str(out), str(tmp_path / f"full{alpha}")) / ck.MODEL))
        assert torch.equal(full["expert.w"], want) and torch.equal(full["vlm.w"], w0["vlm.w"])
