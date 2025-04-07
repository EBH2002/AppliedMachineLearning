import subprocess
import requests
import time
import os

def test_docker():
    """Test the Flask app running inside a Docker container."""
    container_name = "spam_flask_container"

    # Stop and remove existing container if it's running
    subprocess.run(["docker", "stop", container_name], check=False)
    subprocess.run(["docker", "rm", container_name], check=False)

    # Build the Docker image
    subprocess.run(["docker", "build", "-t", "spam-flask-app", "."], check=True)

    # Run the Docker container
    subprocess.run([
        "docker", "run", "-d", "-p", "5000:5000",
        "--name", container_name, "spam-flask-app"
    ], check=True)

    # Wait a few seconds to let the server start
    time.sleep(5)

    # Test the / endpoint
    response = requests.get("http://127.0.0.1:5000/")
    assert response.status_code == 200
    assert "Spam Classifier" in response.text

    # Test the /classify endpoint with a sample spammy message
    test_message = "Free lottery tickets!"
    response = requests.post("http://127.0.0.1:5000/classify", data={"message": test_message})
    assert response.status_code == 200

    data = response.json()
    assert "result" in data and "confidence" in data
    assert isinstance(data["result"], bool)
    assert isinstance(data["confidence"], float)
    assert 0 <= data["confidence"] <= 1

    # Stop and remove the container
    subprocess.run(["docker", "stop", container_name], check=True)
    subprocess.run(["docker", "rm", container_name], check=True)
