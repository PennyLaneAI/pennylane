"""
Module for containing graphql functionality for interacting with the Datasets Service API.
"""

import os
from typing import Any

from requests import post

GRAPHQL_URL = os.getenv("DATASETS_ENDPOINT_URL", "https://cloud.pennylane.ai/graphql")


class GraphQLError(BaseException):
    """Exception for GraphQL"""


def get_graphql(url: str, query: str, variables: dict[str, Any] | None = None):
    """
    Args:
        url: The URL to send a query to.
        query: The main body of the query to be sent.
        variables: Additional input variables to the query body.

    Returns:
        string: json response.
        GraphQLError: if there no response is received or errors are received in the json response.
    """

    json = {"query": query}

    if variables:
        json["variables"] = variables

    response = post(url=url, json=json, timeout=10, headers={"content-type": "application/json"})
    response.raise_for_status()
    if "errors" in response.json():
        all_errors = ",".join(error["message"] for error in response.json()["errors"])
        raise GraphQLError(f"Errors in request: {all_errors}")

    return response.json()


def get_dataset_urls(class_id: str, parameters: dict[str, list[str]]) -> list[tuple[str, str]]:
    """
    Args:
        class_id: Dataset class id e.g 'qchem', 'qspin'
        parameters: Dataset parameters e.g 'molname', 'basis'

    Returns:
        list of tuples (dataset_id, dataset_url)

    Example usage:
    >>> get_dataset_urls("qchem", {"molname": ["H2"], "basis": ["STO-3G"], "bondlength": ["0.5"]})
    [("H2_STO-3G_0.5", "https://cloud.pennylane.ai/datasets/h5/qchem/h2/sto-3g/0.5.h5")]
    """

    response = get_graphql(
        GRAPHQL_URL,
        """
        query GetDatasetsForDownload($datasetClassId: String!, $parameters: [DatasetParameterInput!]) {
          datasetClass(id: $datasetClassId) {
            datasets(parameters: $parameters) {
              id
              downloadUrl
            }
          }
        }
        """,
        {"datasetClassId": class_id, "parameters": parameters},
    )

    return [
        (resp["id"], resp["downloadUrl"]) for resp in response["data"]["datasetClass"]["datasets"]
    ]


def list_data_names() -> list[str]:
    """Get list of dataclass IDs."""
    response = get_graphql(
        GRAPHQL_URL,
        """
        query GetDatasetClasses {
          datasetClasses {
            id
          }
        }
        """,
    )
    return [dsc["id"] for dsc in response["data"]["datasetClasses"]]


def list_attributes(data_name) -> list[str]:
    r"""List the attributes that exist for a specific ``data_name``.

    Args:
        data_name (str): The type of the desired data

    Returns:
        list (str): A list of accepted attributes for a given data name

    .. seealso:: :func:`~.load_interactive`, :func:`~.list_data_names`, :func:`~.load`.

    **Example**

    >>> qml.data.list_attributes(data_name="qchem")
    ['basis_rot_groupings',
     'basis_rot_samples',
     'dipole_op',
     ...
     'vqe_gates',
     'vqe_params']
    """

    response = get_graphql(
        GRAPHQL_URL,
        """
        query ListAttributes($datasetClassId: String!) {
          datasetClass(id: $datasetClassId) {
            attributes {
                name
            }
          }
        }
        """,
        {"datasetClassId": data_name},
    )

    return [attribute["name"] for attribute in response["data"]["datasetClass"]["attributes"]]


def _get_parameter_tree(class_id) -> tuple[list[str], list[str], dict]:
    """Returns the (parameters, attributes, parameter_tree) for a given ``class_id``."""

    response = get_graphql(
        GRAPHQL_URL,
        """
        query GetParameterTree($datasetClassId: String!) {
          datasetClass(id: $datasetClassId) {
            attributes {
              name
            }
            parameters {
              name
            }
            parameterTree
          }
        }
        """,
        {"datasetClassId": class_id},
    )

    parameters = [param["name"] for param in response["data"]["datasetClass"]["parameters"]]
    attributes = [atr["name"] for atr in response["data"]["datasetClass"]["attributes"]]

    return (parameters, attributes, response["data"]["datasetClass"]["parameterTree"])
