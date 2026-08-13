"""Normal forms: what each one keeps, what it drops, and that they are stable."""

import pytest

from warshlab import chars as C
from warshlab import text as T

BASMALA = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
WITH_WAQF = "فَأَصَٰبَتْكُم مُّصِيبَةُ اُ۬لْمَوْتِۖ"


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------


def test_to_raw_strips_tatweel_and_bidi_controls():
    assert T.to_raw("بِســْمِ‏") == "بِسْمِ"


def test_to_raw_collapses_whitespace():
    assert T.to_raw("  قُلْ   هُوَ \n اللَّهُ  ") == "قُلْ هُوَ اللَّهُ"


def test_to_raw_is_idempotent():
    once = T.to_raw(WITH_WAQF)
    assert T.to_raw(once) == once


def test_to_clean_drops_waqf_marks():
    assert "ۖ" not in T.to_clean(WITH_WAQF)


def test_to_clean_keeps_harakat():
    cleaned = T.to_clean(BASMALA)
    assert "ِ" in cleaned and "ْ" in cleaned


def test_to_clean_can_keep_tajweed_marks():
    text = "مِنۢ بَعْدِ"
    assert "ۢ" not in T.to_clean(text)
    assert "ۢ" in T.to_clean(text, drop_tajweed=False)


def test_to_rasm_keeps_only_letters_and_spaces():
    rasm = T.to_rasm(BASMALA)
    assert rasm == "بسم الله الرحمن الرحيم"
    assert all(C.is_letter(ch) or ch == " " for ch in rasm)


def test_to_rasm_is_idempotent():
    once = T.to_rasm(BASMALA)
    assert T.to_rasm(once) == once


def test_unify_letters_folds_alef_variants():
    assert T.unify_letters("أإآٱا") == "ااااا"


def test_to_skeleton_folds_teh_marbuta_and_alef_maksura():
    assert T.to_skeleton("مُصِيبَةُ") == "مصيبه"
    assert T.to_skeleton("عَلَىٰ") == "علي"


def test_to_harakat_keeps_only_marks():
    harakat = T.to_harakat(BASMALA)
    assert harakat
    assert not any(C.is_letter(ch) for ch in harakat)


def test_harakat_and_rasm_are_complementary():
    """Between them the two forms account for every non-space character."""
    source = T.to_clean(BASMALA)
    kept = len(T.to_rasm(source).replace(" ", "")) + len(
        T.to_harakat(source).replace(" ", "")
    )
    assert kept == len([c for c in source if not c.isspace()])


def test_words_splits_on_whitespace():
    assert len(T.words(BASMALA)) == 4
    assert T.words("   ") == []


# --------------------------------------------------------------------------
# Waqf
# --------------------------------------------------------------------------


def test_apply_waqf_ends_with_sukun():
    assert T.apply_waqf(BASMALA).endswith(C.SUKUN)


def test_apply_waqf_replaces_the_final_vowel():
    """The final kasra is dropped, not kept alongside the sukun."""
    result = T.apply_waqf("الرَّحِيمِ")
    assert result == "الرَّحِيمْ"
    assert "ِ" not in result[-2:]


def test_apply_waqf_drops_marks_trailing_the_last_letter():
    result = T.apply_waqf("غَيْرِكُمُۥٓ")
    assert result.endswith("مْ")
    assert "ۥ" not in result
    assert "ٓ" not in result


def test_apply_waqf_removes_pause_marks_first():
    result = T.apply_waqf(WITH_WAQF)
    assert "ۖ" not in result
    assert result.endswith(C.SUKUN)


def test_apply_waqf_is_idempotent():
    once = T.apply_waqf(BASMALA)
    assert T.apply_waqf(once) == once


def test_apply_waqf_does_not_double_the_sukun():
    assert T.apply_waqf("قُلْ").count(C.SUKUN) == 1


def test_apply_waqf_on_empty_input():
    assert T.apply_waqf("") == ""
    assert T.apply_waqf("   ") == ""


def test_apply_waqf_on_text_with_no_letters():
    assert T.apply_waqf("۩۞") == ""


def test_notebook_compat_lets_miniature_letters_carry_the_sukun():
    """The two configurations differ exactly where the original notebook did."""
    source = "غَيْرِكُمُۥ"
    assert T.apply_waqf(source, T.DEFAULT_WAQF).endswith("مْ")
    assert T.apply_waqf(source, T.NOTEBOOK_WAQF).endswith("ۥ" + C.SUKUN)


def test_skip_madd_carriers_leaves_a_bare_final_long_vowel_alone():
    config = T.WaqfConfig(skip_madd_carriers=True)
    assert T.apply_waqf("مُوسَىٰ ٱلْهُدَا", config).endswith("ا")
    assert not T.apply_waqf("مُوسَىٰ ٱلْهُدَا", config).endswith(C.SUKUN)


def test_skip_madd_carriers_still_stops_a_vowelled_carrier():
    config = T.WaqfConfig(skip_madd_carriers=True)
    assert T.apply_waqf("هُوَ", config).endswith(C.SUKUN)


def test_teh_marbuta_to_heh():
    config = T.WaqfConfig(teh_marbuta_to_heh=True)
    assert T.apply_waqf("مُصِيبَةُ", config).endswith("هْ")
    assert T.apply_waqf("مُصِيبَةُ").endswith("ةْ")


def test_waqf_config_rejects_unknown_carrier_set():
    with pytest.raises(ValueError, match="sukun_carriers"):
        T.WaqfConfig(sukun_carriers="everything").carriers()


def test_ends_with_sukun():
    assert T.ends_with_sukun("قُلْ")
    assert not T.ends_with_sukun("قُلُ")
    assert T.ends_with_sukun(T.apply_waqf(BASMALA))


# --------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------


@pytest.mark.parametrize("form", T.FORMS)
def test_every_form_runs_and_returns_a_string(form):
    assert isinstance(T.normalize(WITH_WAQF, form), str)


@pytest.mark.parametrize("form", T.FORMS)
def test_every_form_handles_empty_input(form):
    assert T.normalize("", form) == ""


def test_normalize_rejects_unknown_form():
    with pytest.raises(ValueError, match="unknown form"):
        T.normalize(BASMALA, "sideways")


def test_normalize_defaults_to_waqf():
    assert T.normalize(BASMALA) == T.apply_waqf(BASMALA)
