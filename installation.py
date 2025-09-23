import os
import subprocess
import sys

# ✅ 1. Enforce Python 3.11 only
if not (sys.version_info.major == 3 and sys.version_info.minor == 11):
    print(f"❌ This project requires Python 3.11.x, but you are using {sys.version.split()[0]}")
    sys.exit(1)

def run_command(cmd, cwd=None):
    """Run shell commands with error handling."""
    print(f"\n>>> Running: {cmd} (cwd={cwd or os.getcwd()})")
    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True, cwd=cwd
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)
        sys.exit(1)

# ✅ 2. Install requirements
repo_path = os.path.dirname(os.path.abspath(__file__))
requirements_file = os.path.join(repo_path, "requirements.txt")
# ✅ 2a. Create virtual environment if not exists
venv_path = os.path.join(repo_path, ".venv")
if not os.path.exists(venv_path):
    run_command(f"{sys.executable} -m venv .venv", cwd=repo_path)
    print("✅ Created virtual environment at .venv")
else:
    print("✅ Virtual environment already exists at .venv")

run_command(f"source {venv_path}/bin/activate", repo_path)
if os.path.exists(requirements_file):
    run_command(f"uv pip install -r requirements.txt", cwd=repo_path)
else:
    print("⚠️ No requirements.txt found in repo.")

# ✅ 3. Check or create .env file
env_file = os.path.join(repo_path, ".env")

default_env_content = """OPENAI_API_KEY="OPENAI_API_KEY"
OPENAI_MODEL="OPENAI_MODEL"
OPENAI_TEMPERATURE=OPENAI_TEMPERATURE
ENTERPRISE_GRAPHQL_URL=FOR_PDM_KEY_CONTACT_RIVERSTONE_SUPPORT_TEAM
ENTERPRISE_PRISE_GRAPHQL_URL=FOR_PDM_KEY_CONTACT_RIVERSTONE_SUPPORT_TEAM
ENTERPRISE_API_KEY=FOR_PDM_KEY_CONTACT_RIVERSTONE_SUPPORT_TEAM
URL=FOR_PDM_KEY_CONTACT_RIVERSTONE_SUPPORT_TEAM
EMAIL_HOST="smtp.gmail.com"
EMAIL_PORT=587
EMAIL_USER="YOUR_EMAIL_ADDRESS"
EMAIL_PASS="EMAIL_PASS_KEY"
"""

if not os.path.exists(env_file):
    with open(env_file, "w") as f:
        f.write(default_env_content)
    print(f"⚠️  .env file was missing, created with placeholder values at:\n   {env_file}")
    print("➡️  Please update it with valid credentials.")
    sys.exit(1)

# ✅ 4. Validate .env contents
required_values = {
    "OPENAI_API_KEY": "OPENAI_API_KEY",
    "OPENAI_MODEL": "OPENAI_MODEL",
    "OPENAI_TEMPERATURE": "OPENAI_TEMPERATURE",
    "ENTERPRISE_GRAPHQL_URL": "FOR_PDM_KEY_CONTACT_RIVERSTONE_SUPPORT_TEAM",
    "ENTERPRISE_PRISE_GRAPHQL_URL": "FOR_PDM_KEY_CONTACT_RIVERSTONE_SUPPORT_TEAM",
    "ENTERPRISE_API_KEY": "FOR_PDM_KEY_CONTACT_RIVERSTONE_SUPPORT_TEAM",
    "URL": "FOR_PDM_KEY_CONTACT_RIVERSTONE_SUPPORT_TEAM",
    "EMAIL_USER": "YOUR_EMAIL_ADDRESS",
    "EMAIL_PASS": "EMAIL_PASS_KEY",
}

found = {}
with open(env_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip('"')
            found[key] = value

invalid = []
for key, bad_value in required_values.items():
    if key not in found:
        invalid.append(f"{key} is missing")
    elif found[key] == bad_value:
        invalid.append(f"{key} still has placeholder value: {bad_value}")

if invalid:
    print("❌ Invalid .env file:")
    for item in invalid:
        print("   -", item)
    print("\n⚠️ Please update your .env file with correct values before continuing.")
    sys.exit(1)
else:
    print("✅ .env file validated successfully.")

# ✅ 5. Run uv command inside each tool folder
tools = []
for root, dirs, files in os.walk(repo_path):
    for dir_name in dirs:
        tool_path = os.path.join(root, dir_name)
        main_file = os.path.join(tool_path, "main.py")

        if "_tool" in dir_name.lower() and os.path.isfile(main_file):
            tools.append(tool_path)

if not tools:
    print("⚠️ No tool directories with main.py found in the repo.")
else:
    for tool_path in tools:
        run_command("uv run mcp install main.py", cwd=tool_path)


print("\n🎉 Installation finished successfully!")
