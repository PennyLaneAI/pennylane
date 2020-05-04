# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
APIClient module
================

**Module name:** :mod:`pennylane.api_client`

.. currentmodule:: pennylane.api_client


This module provides a thin client that communicates with remote APIs over the HTTP
protocol, based on the requests module. It also provides helper classes to facilitate interacting
with this API via the Resource subclasses, as well as the ResourceManager wrapper around APIClient
that is available for each resource.

A single :class:`~.APIClient` instance can be used throughout one's session in the application.
The application will attempt to configure the :class:`~.APIClient` instance using a configuration
file or defaults, but the user can choose to override various parameters of the :class:`~.APIClient`
manually.

Classes
-------

.. autosummary::
   APIClient
   Resource
   ResourceManager
   Field
   Job
   JobResult
   JobCircuit

Exceptions
----------

.. autosummary::
   MethodNotSupportedException
   ObjectAlreadyCreatedException
   JobNotQueuedError
   JobExecutionError

----
"""
import urllib
import json
import warnings
import os

import dateutil.parser

import requests


def join_path(base_path, path):
    """
    Joins two paths, a base path and another path and returns a string.

    Args:
        base_path (str): The left side of the joined path.
        path (str): The right side of the joined path.

    Returns:
        str: A joined path.
    """
    return urllib.parse.urljoin("{}/".format(base_path), path)


class MethodNotSupportedException(TypeError):
    """
    Raised when a ResourceManager method is not supported for a
    particular Resource.
    """


class ObjectAlreadyCreatedException(TypeError):
    """
    Raised when an object has already been created but the user
    is attempting to create it again.
    """


class JobNotQueuedError(Exception):
    """
    Raised when a job is not successfully queued for whatever reason.
    """


class JobExecutionError(Exception):
    """
    Raised when job execution failed and a job result does not exist.
    """


class APIClient:
    """
    Allows the user to connect to the remote API.

    Keyword Args:
        hostname (str): cloud platform hostname
        api_key (str): cloud platform API key
    """

    USER_AGENT = "pennylane-api-client/0.1"

    def __init__(self, **kwargs):
        self.HOSTNAME = kwargs.get("hostname", None)
        self.BASE_URL = "https://{}".format(self.HOSTNAME)
        self.AUTHENTICATION_TOKEN = os.getenv("API_KEY") or kwargs.get("api_key", None)
        self.DEBUG = False

        if "DEBUG" in os.environ:
            # if provided, get debug mode from environment variable
            self.DEBUG = json.loads(os.getenv("DEBUG").lower())

        # keyword argument overwrites DEBUG environment variable
        self.DEBUG = kwargs.get("debug", self.DEBUG)

        self.HEADERS = {"User-Agent": self.USER_AGENT}

        if self.AUTHENTICATION_TOKEN is not None:
            self.set_authorization_header(self.AUTHENTICATION_TOKEN)
        else:
            raise PermissionError("API key must be provided")

        if self.DEBUG:
            self.errors = []
            self.responses = []

    def set_authorization_header(self, authentication_token, header_key="Authorization"):
        """
        Adds the authorization header to the headers dictionary to be included
        with all API requests.

        Args:
            authentication_token (str): an authentication token used to access the API

        Kwargs:
            header_key (str): key to be used in header which has authentication_token as its value
        """
        self.HEADERS[key] = authentication_token

    def join_path(self, path):
        """
        Joins a base url with an additional path (e.g., a resource name and ID).

        Args:
            path (str): A path to be joined with ``BASE_URL``

        Returns:
            str: resulting joined path
        """
        return join_path(self.BASE_URL, path)

    def request(self, method, **params):
        """
        Calls ``method`` with ``params`` after applying headers. Records the request type and
        parameters to ``self.errors`` if the request is not successful, and the response to
        ``self.responses`` if a response is returned from the server.

        Args:
            method: one of ``requests.get`` or ``requests.post``
            **params: the parameters to pass on to the method (e.g. ``url``, ``data``, etc.)

        Returns:
            requests.Response: a response object, or None if no response could be fetched
        """
        supported_methods = (requests.get, requests.post)
        if method not in supported_methods:
            raise TypeError("Unexpected or unsupported method provided")

        params["headers"] = self.HEADERS

        try:
            response = method(**params)
        except Exception as e:
            if self.DEBUG:
                self.errors.append((method, params, e))
            raise

        if self.DEBUG:
            self.responses.append(response)

        return response

    def get(self, path):
        """
        Sends a GET request to the provided path. Returns a response object.

        Args:
            path (str): path to send the GET request to

        Returns:
            requests.Response: A response object, or None if no response could be fetched
        """
        return self.request(requests.get, url=self.join_path(path))

    def post(self, path, payload):
        """
        Converts payload to a JSON string. Sends a POST request to the provided
        path. Returns a response object.

        Args:
            path (str): path to send the GET request to
            payload: JSON serializable object to be sent to the server

        Returns:
            requests.Response: A response object, or None if no response could be fetched
        """
        return self.request(requests.post, url=self.join_path(path), data=json.dumps(payload))


class ResourceManager:
    """
    Handles all interactions with APIClient by the Resource.
    """

    http_response_data = None
    http_response_status_code = None
    errors = None

    def __init__(self, resource, client=None):
        """
        Initialize the manager with resource and client instances. A client
        instance is used as a persistent HTTP communications object, and a
        resource instance corresponds to a particular type of resource (e.g.,
        Job)
        """
        self.resource = resource
        self.client = client or APIClient()
        self.errors = []

    def join_path(self, path):
        """
        Joins a resource base path with an additional path (e.g., an ID)
        """
        return join_path(self.resource.PATH, path)

    def get(self, resource_id=None):
        """
        Attempts to retrieve a particular record by sending a GET
        request to the appropriate endpoint. If successful, the resource
        object is populated with the data in the response.

        Args:
            resource_id (int): the ID of an object to be retrieved
        """
        if "GET" not in self.resource.SUPPORTED_METHODS:
            raise MethodNotSupportedException("GET method on this resource is not supported")

        if resource_id is not None:
            response = self.client.get(self.join_path(str(resource_id)))
        else:
            response = self.client.get(self.resource.PATH)
        self.handle_response(response)

    def create(self, **params):
        """
        Attempts to create a new instance of a resource by sending a POST
        request to the appropriate endpoint.

        Args:
            **params: arbitrary parameters to be passed on to the POST request
        """
        if "POST" not in self.resource.SUPPORTED_METHODS:
            raise MethodNotSupportedException("POST method on this resource is not supported")

        if self.resource.id:
            raise ObjectAlreadyCreatedException("ID must be None when calling create")

        response = self.client.post(self.resource.PATH, params)

        self.handle_response(response)

    def handle_response(self, response):
        """
        Store the status code on the manager object and handle the response
        based on the status code.

        Args:
            response (requests.Response): a response object to be parsed
        """
        if hasattr(response, "status_code"):
            self.http_response_data = response.json()
            self.http_response_status_code = response.status_code

            if response.status_code in (200, 201):
                self.handle_success_response(response)
            else:
                self.handle_error_response(response)
        else:
            self.handle_no_response()

    def handle_no_response(self):
        """
        Placeholder method to handle an unsuccessful request (e.g. due to no network connection).
        """
        warnings.warn("Your request could not be completed")

    def handle_success_response(self, response):
        """
        Handles a successful response by refreshing the instance fields.

        Args:
            response (requests.Response): a response object to be parsed
        """
        self.refresh_data(response.json())

    def handle_error_response(self, response):
        """
        Handles an error response that is returned by the server.

        Args:
            response (requests.Response): a response object to be parsed
        """

        error = {"status_code": response.status_code, "content": response.json()}
        self.errors.append(error)
        try:
            response.raise_for_status()
        except Exception as e:
            raise Exception(response.text) from e

    def refresh_data(self, data):
        """
        Refreshes the instance's attributes with the provided data and
        converts it to the correct type.

        Args:
            data (dict): A dictionary containing keys and values of data to be stored on the object.
        """
        for field in self.resource.fields:
            field.set(data.get(field.name, None))

        if hasattr(self.resource, "refresh_data"):
            self.resource.refresh_data()


class Resource:
    """
    Base class for an API resource. Should be extended for each resource endpoint.
    """

    SUPPORTED_METHODS = ()
    PATH = ""
    fields = ()

    def __init__(self, client=None):
        """
        Initialize the Resource by populating attributes based on fields and setting a manager.

        Args:
            client (APIClient): An APIClient instance to use as a client.
        """
        self.manager = ResourceManager(self, client=client)
        for field in self.fields:
            setattr(self, field.name, field)

    def reload(self):
        """
        A helper method to fetch the latest data from the API.
        """
        if not hasattr(self, "id"):
            raise TypeError("Resource does not have an ID")

        if self.id:
            self.manager.get(self.id.value)
        else:
            warnings.warn("Could not reload resource data", UserWarning)


class Field:
    """
    Classifies and cleans data returned by the API.
    """

    value = None

    def __init__(self, name, clean=str):
        """
        Initialize the Field object with a name and a cleaning function.

        Args:
            name (str): A string representing the name of the field (e.g., "created_at").
            clean: A method that returns a cleaned value of the field, of the correct type.
        """
        self.name = name
        self.clean = clean

    def __repr__(self):
        """
        Return the string representation of the value.
        """
        return "<{} {}: {}>".format(self.name, self.__class__.__name__, str(self.value))

    def __bool__(self):
        """
        Use the value to determine boolean state.
        """
        return self.value is not None

    def set(self, value):
        """
        Set the value of the Field to `value`.

        Args:
            value: The value to be stored on the Field object.
        """
        self.value = value

    @property
    def cleaned_value(self):
        """
        Return the cleaned value of the field (for example, an integer or Date
        object)
        """
        return self.clean(self.value) if self.value is not None else None


class Job(Resource):
    """
    API resource corresponding to jobs.
    """

    SUPPORTED_METHODS = ("GET", "POST")
    PATH = "jobs"

    def __init__(self, client=None):
        """
        Initialize the Job resource with a set of pre-defined fields.
        """
        self.fields = (
            Field("id", str),
            Field("type", str),
            Field("status", str),
            Field("request", dateutil.parser.parse),
            Field("response", dateutil.parser.parse),
            Field("data"),
        )

        self.result = None
        self.circuit = None

        super().__init__(client=client)

    @property
    def is_complete(self):
        """
        Returns True if the job status is "COMPLETE". Case insensitive. Returns False otherwise.
        """
        return self.status.value and self.status.value.upper() == "COMPLETED"

    @property
    def is_failed(self):
        """
        Returns True if the job status is "FAILED". Case insensitive. Returns False otherwise.
        """
        return self.status.value and self.status.value.upper() == "FAILED"

    def refresh_data(self):
        """
        Refresh the job fields and attach a JobResult and JobCircuit object to the Job instance.
        """
        if self.result is None:
            self.result = JobResult(self.id.value, client=self.manager.client)

        if self.circuit is None:
            self.circuit = JobCircuit(self.id.value, client=self.manager.client)


class JobResult(Resource):
    """
    API resource corresponding to the job result.
    """

    SUPPORTED_METHODS = ("GET",)
    PATH = "v0/jobs/{job_id}/data"

    def __init__(self, job_id, client=None):
        """
        Initialize the JobResult resource with a pre-defined field.

        Args:
            job_id (int): The ID of the Job object corresponding to the JobResult object.
        """
        self.fields = (Field("result", json.loads),)

        self.PATH = self.PATH.format(job_id=job_id)
        super().__init__(client=client)


class JobCircuit(Resource):
    """
    API resource corresponding to the job circuit.
    """

    SUPPORTED_METHODS = ("GET",)
    PATH = "v0/jobs/{job_id}/circuit"

    def __init__(self, job_id, client=None):
        """
        Initialize the JobCircuit resource with a pre-defined field.

        Args:
            job_id (int): The ID of the Job object corresponding to the JobResult object.
        """
        self.fields = (Field("circuit"),)

        self.PATH = self.PATH.format(job_id=job_id)
        super().__init__(client=client)
