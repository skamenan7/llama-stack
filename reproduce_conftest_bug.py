import pytest
import sys


def test_args(pytestconfig):
    args = pytestconfig.invocation_params.args
    print(f"DEBUG: args={args}")

    running_unit = any("tests/unit" in str(arg) or "tests\\unit" in str(arg) for arg in args)

    if not args or args == (".",) or args == ("tests",) or args == ("tests/",):
        running_unit = True

    print(f"DEBUG: running_unit={running_unit}")
