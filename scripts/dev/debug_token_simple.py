from huggingface_hub import login
import os

token = "YOUR_HF_TOKEN"
print(f"Token length: {len(token)}")
print(f"Token repr: {repr(token)}")

try:
    # Try strict login first
    login(token=token, add_to_git_credential=False)
    print("Login success!")
except Exception as e:
    print(f"Login failed: {e}")
