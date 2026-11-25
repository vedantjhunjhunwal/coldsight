"""GS1 mod-10 check digit, used on pallet IDs so misreads can be rejected."""


def check_digit(body: str) -> str:
    total = 0
    for i, ch in enumerate(reversed(body)):
        total += int(ch) * (3 if i % 2 == 0 else 1)
    return str((10 - total % 10) % 10)


def make_id(body: str) -> str:
    return body + check_digit(body)


def is_valid(code: str) -> bool:
    return len(code) == 8 and code.isdigit() and check_digit(code[:-1]) == code[-1]
