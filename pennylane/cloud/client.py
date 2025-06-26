"""The PL cloud client."""
import ast
import re

import cloudpickle
import requests


class Client:

    def __init__(self, base_url):
        self._base_url = base_url

    def submit(self, qnode):
        """Submit a circuit to the cloud."""

        data = cloudpickle.dumps(qnode)

        try:
            response = requests.post(self._base_url + "/start-job", data=data, headers={"Content-Type": "application/octet-stream"})
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            return response.json().get("job_uuid")

        except requests.exceptions.RequestException as e:
            print(f"❌ An error occurred sending job: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"Server response: {e.response.text}")

    def get_result(self, job_uuid):
        """Get the result of a job."""

        try:
            response = requests.get(f"{self._base_url}/get-results/{job_uuid}")
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            raw_result = response.json().get("results")
            match = re.search(r"data\s*=\s*\n?(\[.*?\])", raw_result, re.DOTALL)
            data_str = match.group(1)
            return ast.literal_eval(data_str)

        except requests.exceptions.RequestException as e:
            print(f"❌ An error occurred getting job status: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"Server response: {e.response.text}")
