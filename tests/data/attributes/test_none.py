from pennylane.data.attributes.none import DatasetNone


class TestDatasetsNone:
    def test_value_init(self):
        """Test that DatasetsNone can be value-initialized."""

        dsets_none = DatasetNone(None)

        assert dsets_none.get_value() is None
        assert bool(dsets_none) is False
        assert dsets_none.info["type_id"] == "none"
        assert dsets_none.bind.shape == ()

    def test_bind_init(self):
        """Test that DatasetsNone can be bind-initialized."""
        bind = DatasetNone(None).bind

        dsets_none = DatasetNone(bind=bind)

        assert dsets_none.get_value() is None
        assert bool(dsets_none) is False
        assert dsets_none.info["type_id"] == "none"
        assert dsets_none.bind.shape == ()
