"""run_batch's PASS must mean the answer is CORRECT, not merely 'answered'."""
import run_batch


def test_normalisation_ignores_commas_currency_and_spaces():
    assert run_batch.normalise("Rs 74,671.29 crore (₹ 74 671.29)") == "rs74671.29crore(₹74671.29)"


def test_contains_all_expected_values():
    ans = "Net profit was ₹74,671.29 crore for FY2026."
    assert run_batch.is_correct(ans, ["74671.29"])
    assert not run_batch.is_correct("Net profit was Rs 3,512.96 crore.", ["74671.29"])   # the real wrong answer
    assert run_batch.is_correct("1.24% then 1.15%", ["1.24", "1.15"])
    assert not run_batch.is_correct("1.15% only", ["1.24", "1.15"])


def test_no_expectation_means_any_answer_is_acceptable():
    assert run_batch.is_correct("anything", [])
    assert run_batch.is_correct("anything", None)


def test_every_answerable_value_question_declares_what_it_must_contain():
    by_num = {q[0]: q for q in run_batch.QUESTIONS}
    assert by_num[1][3] == ["74671.29"] and by_num[2][3] == ["19.71"] and by_num[17][3] == ["79219.46"]
