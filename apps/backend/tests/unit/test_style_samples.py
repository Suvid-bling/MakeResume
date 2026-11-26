"""Style sample registry and profile parsing."""

import re

import pytest

from app.prompts.style import STYLE_SAMPLES, get_style_sample
from app.services.style_profile import bullet_targets, compute_style_profile

_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def test_sample_ids_are_unique_and_well_formed() -> None:
    ids = [s.id for s in STYLE_SAMPLES]
    assert len(ids) == len(set(ids))
    for sample_id in ids:
        assert _ID_RE.match(sample_id)


def test_every_registered_sample_has_nonempty_text_and_label() -> None:
    for sample in STYLE_SAMPLES:
        assert sample.text.strip()
        assert sample.label.strip()
        assert sample.language.strip()


def test_get_style_sample_known_and_unknown() -> None:
    first = STYLE_SAMPLES[0]
    assert get_style_sample(first.id) is first
    assert get_style_sample("does-not-exist") is None


def test_every_registered_sample_parses_into_a_profile() -> None:
    """Catches a malformed sample before it ships (per the style.py docstring)."""
    for sample in STYLE_SAMPLES:
        profile = compute_style_profile(sample)
        assert profile.bullets_per_entry, f"{sample.id}: no entries parsed"
        assert profile.sample_bullets, f"{sample.id}: no bullets parsed"


def test_senior_fullstack_zh_profile_matches_known_shape() -> None:
    sample = get_style_sample("senior_fullstack_zh")
    assert sample is not None
    profile = compute_style_profile(sample)

    assert profile.language == "zh"
    assert profile.bullets_per_entry == (4, 3, 1)
    assert profile.quantified_ratio == pytest.approx(1.0)
    assert profile.summary_sentences == 2
    assert profile.skills_grouped is True
    assert profile.skills_separator == "、"
    assert profile.bullet_length_min > 0
    assert profile.bullet_length_max >= profile.bullet_length_min

    for name in (
        "Daniel Okafor",
        "Brightmoor Commerce",
        "Northgate Studio",
        "Ferrule Labs",
    ):
        assert any(name.lower() in term.lower() for term in profile.protected_terms), (
            name
        )

    assert any("2,400" in n or "2400" in n for n in profile.distinctive_numbers) or any(
        "31%" in n for n in profile.distinctive_numbers
    )


def test_bullet_targets_maps_proportionally_with_cap() -> None:
    sample = get_style_sample("senior_fullstack_zh")
    assert sample is not None
    profile = compute_style_profile(sample)

    assert bullet_targets(profile, 1, cap=None) == [4]
    assert bullet_targets(profile, 3, cap=None) == [4, 3, 1]
    assert bullet_targets(profile, 5, cap=None)[0] == 4
    assert bullet_targets(profile, 5, cap=None)[-1] == 1
    capped = bullet_targets(profile, 3, cap=2)
    assert capped == [2, 2, 1]


def test_bullet_targets_empty_entries() -> None:
    sample = get_style_sample("senior_fullstack_zh")
    assert sample is not None
    profile = compute_style_profile(sample)
    assert bullet_targets(profile, 0, cap=None) == []
