"""
Fetch IQM Resonance calibration data and save to JSON.
Requires: pip install iqm-client
Set IQM_TOKEN environment variable with your Resonance API token.
"""
import os
import json
import requests
from iqm.iqm_client import IQMClient

# ─── CONFIG ───
IQM_SERVER_URL = "https://resonance.iqm.tech"
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "data/iqm_calibration.json")

def get_calibration_data(client, calibration_set_id=None, filename=None):
    headers = {"User-Agent": client._signature}
    bearer_token = client._token_manager.get_bearer_token()
    headers["Authorization"] = bearer_token

    if calibration_set_id:
        url = os.path.join(client._api.iqm_server_url,
                           "calibration/metrics/", calibration_set_id)
    else:
        url = os.path.join(client._api.iqm_server_url,
                           "calibration/metrics/latest")

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()

    if filename:
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Saved calibration to {filename}")

    return data

if __name__ == "__main__":
    token = os.environ.get("IQM_TOKEN")
    if not token:
        print("ERROR: set IQM_TOKEN environment variable with your Resonance token")
        sys.exit(1)

    client = IQMClient(IQM_SERVER_URL, quantum_computer="emerald")
    data = get_calibration_data(client, filename=OUTPUT_PATH)

    # Print available metrics so we know what we got
    print("\nAvailable metrics:")
    for metric_key in data["metrics"].keys():
        print(" ", metric_key)
