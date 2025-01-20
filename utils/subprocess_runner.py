import json
import subprocess


def run_command(command):
    """
    Executes a shell command and returns its output.

    Args:
        command (str): The shell command to execute.

    Returns:
        str: Output of the command.

    Raises:
        RuntimeError: If the command fails.
    """
    with subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1
    ) as sp:
        stdout, stderr = sp.communicate()
        if sp.returncode != 0:
            raise RuntimeError(f"Command failed: {stderr.decode('utf-8')}")
        return stdout.decode("utf-8")


def make_api_request(endpoint, payload):
    """
    Sends a request to the API and handles the response.

    Args:
        endpoint (str): API endpoint.
        payload (dict): Payload to send in the request.

    Returns:
        dict: Parsed JSON response.

    Raises:
        RuntimeError: If the API returns an error.
        json.JSONDecodeError: If the response cannot be parsed as JSON.
    """
    try:
        escaped_payload = json.dumps(payload).replace('"', '\\"')
        command = f'curl -s {endpoint} -d "{escaped_payload}"'
        raw_response = run_command(command)
        return json.loads(raw_response)
    except json.JSONDecodeError as e:
        print(f"Failed to parse response: {raw_response}")
        raise e
