"""vis_noise_candidates 纯逻辑测试：tier 门槛分级 + 全局贪心选样（无需 GPU/数据）。"""

from scripts.exp.vis_noise_candidates import _pick, _tier


def _c(idx, tier, margin):
    return {"idx": idx, "tier": tier, "margin": margin}


def test_tier_low_sigma_prefers_high_m3_with_margin():
    # σ≤0.3：A 需 m3≥0.93 且对 M1/M2 各自 margin≥0.10
    assert _tier((0.70, 0.70, 0.96), 0.9, 1.0, 0.2) == "A"
    # margin 恰 0.10 边界含取（0.93 ≥ 0.83+0.10）
    assert _tier((0.83, 0.83, 0.93), 0.9, 1.0, 0.3) == "A"
    # m3 够高但 margin 不足 → 降 B（m3≥0.85 且 margin≥0.05）
    assert _tier((0.88, 0.88, 0.93), 0.9, 1.0, 0.1) == "B"


def test_tier_low_sigma_boundaries():
    # m3=0.93 边界含取；m3=0.92 → 降 B
    assert _tier((0.5, 0.5, 0.93), 1.0, 1.0, 0.0) == "A"
    assert _tier((0.5, 0.5, 0.92), 1.0, 1.0, 0.0) == "B"
    # B 边界：m3=0.85、margin 恰 0.05 含取
    assert _tier((0.79, 0.79, 0.85), 1.0, 1.0, 0.1) == "B"
    # m3 不足 B（0.84）但 M3' 严格最优 → C
    assert _tier((0.5, 0.5, 0.84), 1.0, 1.0, 0.2) == "C"
    # M3' 非最优 → None
    assert _tier((0.9, 0.5, 0.84), 1.0, 1.0, 0.2) is None


def test_tier_sigma_05_usable_guards():
    # σ=0.5 tier A：margin≥0.10 + coh≥0.7 + size_ratio∈[0.3,2.5]
    assert _tier((0.10, 0.10, 0.45), 0.8, 1.0, 0.5) == "A"
    # coherence 不足 → B
    assert _tier((0.10, 0.10, 0.45), 0.6, 1.0, 0.5) == "B"
    # size_ratio 越界 → B
    assert _tier((0.10, 0.10, 0.45), 0.8, 2.6, 0.5) == "B"
    # coh<0.5 → C（仅 M3' 最优）
    assert _tier((0.10, 0.10, 0.45), 0.4, 1.0, 0.5) == "C"
    # M3' 非最优 → None
    assert _tier((0.6, 0.1, 0.45), 0.8, 1.0, 0.5) is None


def test_pick_global_dedup_and_hardest_first():
    cands = {
        0.5: [_c(1, "A", 0.30), _c(2, "A", 0.25)],
        0.0: [_c(1, "A", 0.20), _c(4, "A", 0.10)],
        0.3: [_c(2, "A", 0.20), _c(3, "A", 0.05)],
    }
    picked = _pick(cands)
    # 难度优先（σ 降序）：0.5 先选 1/2；0.3 拿 3；0.0 只剩 4
    assert picked[0.5] == [_c(1, "A", 0.30), _c(2, "A", 0.25)]
    assert [c["idx"] for c in picked[0.3]] == [3]
    assert [c["idx"] for c in picked[0.0]] == [4]


def test_pick_fills_from_lower_tiers():
    cands = {
        0.5: [_c(1, "A", 0.30)] + [_c(i, "C", 0.01) for i in range(2, 7)],
        0.0: [_c(1, "A", 0.20), _c(10, "A", 0.15)],
    }
    picked = _pick(cands)
    assert [c["tier"] for c in picked[0.5]] == ["A", "C", "C", "C", "C"]
    # img1 被 0.5 占用 → 0.0 只剩 img10
    assert [c["idx"] for c in picked[0.0]] == [10]


def test_pick_caps_at_five():
    cands = {0.1: [_c(i, "A", 0.5) for i in range(10)]}
    picked = _pick(cands)
    assert len(picked[0.1]) == 5
    assert len({c["idx"] for cs in picked.values() for c in cs}) == 5
