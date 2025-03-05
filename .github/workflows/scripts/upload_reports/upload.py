import requests
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import json


class PLOSSSettings(BaseSettings):
    """Settings for the PennyLane OSS service."""

    model_config = SettingsConfigDict(env_prefix="PENNYLANE_OSS_SERVER_")

    #TODO: Remove _dev suffixes before merging
    endpoint_url_dev: str
    """URL of the PennyLane OSS Service."""

    api_key_dev: str
    """API key to be supplied to the PennyLane OSS Service."""


def read_reports() -> dict:
    """Read the reports from the reports directory."""

    reports = [str(p) for p in Path(".github/test-reports").glob("*.xml")]
    print(f"Found {len(reports)} report files in .github/test-reports/")

    report_contents = {}
    for report_path in reports:
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                report_contents[Path(report_path).name] = f.read()
                print(f"Successfully read report: {Path(report_path).name}")
        except Exception as e:
            print(f"Warning: Failed to read report {report_path}: {str(e)}")
            continue

    if not report_contents:
        print("Warning: No report contents were read")

    return report_contents


def upload_reports(report_contents: dict):
    """Upload a report to the PennyLane OSS Service."""

    settings = PLOSSSettings()

    payload = json.dumps(report_contents)
    headers = {"Authorization": f"Bearer {settings.api_key_dev}"}

    try:
        print(f"Attempting to upload reports to {settings.endpoint_url_dev}")
        response = requests.post(
            settings.endpoint_url_dev,
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        print(f"Successfully uploaded reports. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Warning: Failed to upload reports: {str(e)}")


if __name__ == "__main__":
    report_contents = read_reports()
    upload_reports(report_contents)
