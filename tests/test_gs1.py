from coldsight.gs1 import check_digit, is_valid, make_id


def test_known_check_digit():
    # GS1 example: body 629104150021 -> check digit 3 (mod-10, weights 3,1 from the right)
    assert check_digit("629104150021") == "3"


def test_every_single_digit_error_is_caught():
    code = make_id("4012345")
    assert is_valid(code)
    for i in range(8):
        for d in "0123456789":
            if d != code[i]:
                assert not is_valid(code[:i] + d + code[i + 1:])


def test_rejects_wrong_length_and_letters():
    assert not is_valid("1234567")
    assert not is_valid("40I23457")
