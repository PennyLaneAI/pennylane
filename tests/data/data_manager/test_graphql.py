"""Basic integration tests for the Datasets Service API"""

import pytest
import requests

import pennylane.data.data_manager
from pennylane.data.data_manager.graphql import GRAPHQL_URL


# pylint: disable=protected-access
class TestGetGraphql:
    """Tests for the ``get_graphql()`` function."""

    query = """
        query DatasetClass {
          datasetClasses {
            id
          }
        }
        """

    def test_return_json(self):
        """Tests that a dictionary representation of a json response is returned for a
        valid query and url."""
        response = pennylane.data.data_manager.graphql.get_graphql(
            GRAPHQL_URL,
            self.query,
        )
        assert isinstance(response, dict)

    def test_bad_url(self):
        """Tests that a ``ConnectionError`` is returned for a valid query and invalid url."""

        with pytest.raises(requests.exceptions.ConnectionError):
            pennylane.data.data_manager.graphql.get_graphql(
                "https://bad/url/graphql",
                self.query,
            )

    def test_bad_query(self):
        """Tests that a ``HTTPError`` is returned for a invalid query and valid url."""

        bad_query = """
            query BadQuery {
              badQuery {
                  field
              }
            }
            """

        with pytest.raises(requests.exceptions.HTTPError):
            pennylane.data.data_manager.graphql.get_graphql(
                GRAPHQL_URL,
                bad_query,
            )
