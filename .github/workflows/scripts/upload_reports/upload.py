from argparse import ArgumentParser, Namespace
import requests
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import json
import os
import warnings


class PLOSSSettings(BaseSettings):
    """Settings for the PennyLane OSS service."""

    model_config = SettingsConfigDict(env_prefix="PENNYLANE_OSS_SERVER_")

    endpoint_url: str
    """URL of the PennyLane OSS Service."""

    api_key: str
    """API key to be supplied to the PennyLane OSS Service."""


def read_reports(workspace_path: Path | None) -> dict:
    """Read the reports from the reports directory."""

    # Look for reports in the absolute path where GitHub Actions writes them
    print(f"GitHub workspace: {workspace_path}")
    if workspace_path:
        reports_dir = Path(workspace_path) / ".github" / "test-reports"
    else:
        reports_dir = Path(".github/test-reports")
    print(f"Reports directory: {reports_dir}")

    # Look for XML files directly in the test-reports directory
    reports = list(reports_dir.glob("*.xml"))
    print(f"Found {len(reports)} report files")

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
        warnings.warn("No report contents were read")

    return report_contents


def upload_reports(report_contents: dict):
    """Upload a report to the PennyLane OSS Service."""

    settings = PLOSSSettings()
    headers = {"x-api-key": settings.api_key}

    try:
        print(f"Attempting to upload reports to {settings.endpoint_url}")
        response = requests.post(
            settings.endpoint_url,
            headers=headers,
            json=report_contents,
        )
        response.raise_for_status()
        print(f"Successfully uploaded reports. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Warning: Failed to upload reports: {str(e)}")


def parse_args() -> Namespace:
    """Parses the arguments provided to this Python script"""
    parser = ArgumentParser(
        prog="python upload.py",
        description="This module uploads the provided reports to the PennyLane OSS Service.",
    )

    parser.add_argument(
        "--commit-sha",
        type=str,
        required=True,
        help="The SHA of the commit to upload the reports for.",
    )

    parser.add_argument(
        "--branch",
        type=str,
        required=True,
        help="The branch of the commit to upload the reports for.",
    )

    parser.add_argument(
        "--workflow-id",
        type=str,
        required=True,
        help="The ID of the workflow to upload the reports for.",
    )

    parser.add_argument(
        "--workspace-path",
        type=Path,
        required=False,
        default=os.environ.get("GITHUB_WORKSPACE"),
        help="Path to where the repository is cloned to."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report_contents = read_reports(workspace_path=args.workspace_path)

    report_contents["metadata"] = {
        "commit_sha": args.commit_sha,
        "branch": args.branch,
        "workflow_id": args.workflow_id
    }

    upload_reports(report_contents)
