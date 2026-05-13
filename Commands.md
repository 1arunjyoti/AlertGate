# Commands for managing the virtual environment

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
deactivate

# Commands for managing dependencies and auditing for vulnerabilities

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
pip freeze > requirements.txt

python -m pip install --upgrade pip setuptools wheel
python -m pip install pip-audit

pip-audit -r requirements.txt
pip-audit -r requirements.txt --fix

# Commands for managing the Docker environment

docker compose config --quiet # If it prints nothing, your compose file is valid.
docker compose build
docker compose up -d
docker compose up
docker compose down
docker image rm <image-name-or-id>
```
