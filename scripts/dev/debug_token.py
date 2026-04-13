from huggingface_hub import login, HfApi

token = "YOUR_HF_TOKEN"
print(f"Testing token: '{token}'")
try:
    login(token=token, add_to_git_credential=True)
    print("Login successful")
    api = HfApi()
    user = api.whoami(token=token)
    print(f"User info: {user}")
except Exception as e:
    print(f"Login failed: {e}")
