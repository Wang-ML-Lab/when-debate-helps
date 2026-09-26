from when_debate_helps.extraction import answer_is_correct, extract_answer


def test_multiple_choice_prefers_explicit_final_tag() -> None:
    text = "A could be tempting, but calculation gives C. <answer>C</answer>"
    assert extract_answer(text, "multiple_choice") == "C"


def test_multiple_choice_rejects_out_of_range_letter() -> None:
    assert extract_answer("<answer>F</answer>", "multiple_choice", num_choices=4) is None


def test_free_form_uses_last_number_and_normalizes() -> None:
    assert extract_answer("We had 5, then the final value is 1,200.00", "exact") == "1200"
    assert answer_is_correct("1200", "1,200.0", "exact")
