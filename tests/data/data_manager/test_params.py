from pennylane.data.data_manager.params import ParamArg
import pytest


class TestParamArg:
    """Tests for ``ParamArg``."""

    def test_values(self):
        """Tests that ``values()`` returns all the possible values of ``ParamArg``."""

        assert set(ParamArg.values()) == {"full", "default"}

    @pytest.mark.parametrize(
        "val, expect",
        [
            (ParamArg.DEFAULT, True),
            (ParamArg.FULL, True),
            ("full", True),
            ("default", True),
            (None, False),
            ("DEFAULT", False),
        ],
    )
    def test_is_arg(self, val, expect):
        """Tests that ``is_arg()`` returns True iff the argument is a member of ``ParamArg`` or
        one of its values."""
        assert ParamArg.is_arg(val) == expect

    @pytest.mark.parametrize("val", [ParamArg.FULL, ParamArg.DEFAULT])
    def test_str(self, val):
        """Test that __str__ returns the value of the enum."""

        assert str(val) == val.value
