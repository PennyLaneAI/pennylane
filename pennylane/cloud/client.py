"""The PL cloud client."""

import ast
import re
import threading

import cloudpickle
import requests


class Job:

    def __init__(self, client, data):
        self.client = client
        self.data = data
        self.status = "pending"
        self.job_uuid = None
        self._thread = threading.Thread(target=self._submit_job)
        self._thread.start()

    def _submit_job(self):

        try:
            response = requests.post(
                self.client._base_url + "/start-job",
                data=self.data,
                headers={"Content-Type": "application/octet-stream"},
            )
            response.raise_for_status()
            self.job_uuid = response.json().get("job_uuid")
            self.status = "submitted" if self.job_uuid else "failed"
        except requests.exceptions.RequestException as e:
            print(f"❌ An error occurred sending job: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"Server response: {e.response.text}")
            self.status = "failed"

    def get_result(self):
        """Get the result of this job."""
        if not self.job_uuid:
            print("Job has not been submitted or failed to submit.")
            return None
        try:
            response = requests.get(f"{self.client._base_url}/get-results/{self.job_uuid}")
            response.raise_for_status()
            raw_result = response.json().get("results")
            match = re.search(r"data\s*=\s*\n?(\[.*?\])", raw_result, re.DOTALL)
            data_str = match.group(1)
            return ast.literal_eval(data_str)
        except requests.exceptions.RequestException as e:
            print(f"❌ An error occurred getting job status: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"Server response: {e.response.text}")
        except Exception as e:
            print(f"❌ An error occurred parsing job result: {e}")


class Client:

    def __init__(self, base_url):
        self._base_url = base_url

    def submit(self, qnode):
        """Submit a circuit to the cloud asynchronously."""
        data = cloudpickle.dumps(qnode)
        return Job(self, data)
