from pennylane.data.attributes.none import DatasetsNone


class TestDatasetsNone:
    def test_value_init(self):
        """Test that DatasetsNone can be value-initialized."""

        dsets_none = DatasetsNone(None)

        assert dsets_none.get_value() is None
        assert bool(dsets_none) is False
        assert dsets_none.info["type_id"] == "none"
        assert dsets_none.bind.shape == ()

    def test_bind_init(self):
        """Test that DatasetsNone can be bind-initialized."""
        bind = DatasetsNone(None).bind

        dsets_none = DatasetsNone(bind=bind)

        assert dsets_none.get_value() is None
        assert bool(dsets_none) is False
        assert dsets_none.info["type_id"] == "none"
        assert dsets_none.bind.shape == ()
