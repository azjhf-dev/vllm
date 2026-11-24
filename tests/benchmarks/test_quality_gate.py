# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the pure parts of benchmarks/inkling2/quality_gate.py.

Window construction and result comparison run on CPU with no weights, no
GPU and no HF downloads. The vLLM evaluation path is exercised on the 5090.
"""

import json

from benchmarks.inkling2.quality_gate import compare, make_windows


class TestMakeWindows:
    def test_covers_text_with_sequential_stride(self):
        spans = make_windows(list(range(10000)), window=2048, stride=2048,
                             seed=0, max_windows=0)
        assert spans == [(i, i + 2048) for i in range(0, 10000 - 2048 + 1, 2048)]

    def test_deterministic_for_same_seed(self):
        ids = list(range(5000))
        a = make_windows(ids, 1024, 1024, seed=7, max_windows=0)
        b = make_windows(ids, 1024, 1024, seed=7, max_windows=0)
        assert a == b

    def test_max_windows_caps_count(self):
        spans = make_windows(list(range(20000)), 2048, 2048, seed=0,
                             max_windows=3)
        assert len(spans) == 3
        # Sequential output order even though selection was shuffled.
        assert spans == sorted(spans)

    def test_window_longer_than_text_is_empty(self):
        assert make_windows([1, 2, 3], 2048, 2048, 0, 0) == []


class TestCompare:
    def _write(self, tmp_path, name, ppl, fp8):
        d = {"model": "m", "fp8_dense": fp8, "dataset": "ds/cfg",
             "split": "test", "window": 2048, "stride": 2048, "seed": 0,
             "num_windows": 4, "total_scored_tokens": 8000,
             "mean_nll": 1.0, "perplexity": ppl}
        p = tmp_path / name
        p.write_text(json.dumps(d))
        return str(p)

    def test_reports_delta(self, tmp_path, capsys):
        a = self._write(tmp_path, "a.json", 10.0, "off")
        b = self._write(tmp_path, "b.json", 10.04, "lm_head")
        compare(a, b)
        out = capsys.readouterr().out
        assert "+0.400%" in out
        assert "unverified" in out

    def test_mismatched_settings_warn(self, tmp_path, capsys):
        a = self._write(tmp_path, "a.json", 10.0, "off")
        b = json.loads((tmp_path / "a.json").read_text())
        b["window"] = 1024
        (tmp_path / "b.json").write_text(json.dumps(b))
        compare(a, str(tmp_path / "b.json"))
        assert "WARNING" in capsys.readouterr().out
