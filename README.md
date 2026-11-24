# MLOps Project

An end-to-end MLOps pipeline for training, evaluating, and serving machine learning models.

## Features

- ✅ Modular and scalable project structure
- ✅ Reproducible environments using uv
- ✅ Dockerized training and inference
- ✅ CI/CD with GitHub Actions
- ✅ Unit testing
- ✅ Clear data separation (raw vs processed)
- ✅ Model versioning
- ✅ Experiment reproducibility

## Project Structure

```
.
├── .devcontainer/            # Dev container configuration
├── .github/
│   └── workflows/
│       └── tests.yaml        # CI pipeline
├── configs/                  # Configuration files
├── data/
│   ├── raw/                  # Raw datasets
│   └── processed/            # Processed datasets
├── dockerfiles/
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained model artifacts
├── notebooks/                # Jupyter notebooks
├── reports/
│   └── figures/              # Generated plots
├── src/
│   └── project/
│       ├── __init__.py
│       ├── api.py
│       ├── data.py
│       ├── evaluate.py
│       ├── model.py
│       ├── train.py
│       └── visualize.py
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements.txt
├── requirements_dev.txt
└── tasks.py
```

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/konstandinoseng/MLops_Project_Group_49.git
cd your-repo-name
```

## Dependency Management (Using uv)

This project uses `uv` instead of pip for faster dependency management.

### Install uv

**macOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

### Create Virtual Environment

```bash
uv venv
```

**Activate it:**

**macOS / Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
uv pip install -r requirements.txt
```

For development dependencies:

```bash
uv pip install -r requirements_dev.txt
```

## Running the Project

### Training

```bash
python src/project/train.py
```

### Evaluation

```bash
python src/project/evaluate.py
```

### API

```bash
python src/project/api.py
```

## Docker

### Build Training Image

```bash
docker build -f dockerfiles/train.Dockerfile -t project-train .
```

### Build API Image

```bash
docker build -f dockerfiles/api.Dockerfile -t project-api .
```

## Testing

Run all tests:

```bash
pytest
```

## CI/CD

This project uses GitHub Actions for:

- Linting
- Formatting
- Unit tests
- Build checks

Workflow file: `.github/workflows/tests.yaml`

## Data Management

- **Raw data:** `data/raw/`
- **Processed data:** `data/processed/`
- **Models:** `models/`
- **Reports:** `reports/`

These directories should typically not be tracked by Git.

## Pre-commit Hooks

### Install Hooks

```bash
pre-commit install
```

### Run Manually

```bash
pre-commit run --all-files
```

### Automatic Hook Updates

The `Pre-commit auto-update` workflow updates hook versions and opens a pull
request when changes are available. It requests `contents: write` and
`pull-requests: write` for `GITHUB_TOKEN`, but pull request creation also
requires a repository setting:

1. A repository administrator must open **Settings → Actions → General**.
2. Under **Workflow permissions**, enable **Allow GitHub Actions to create and
   approve pull requests** and save. This setting applies to other workflows
   with the necessary permissions too. If it is unavailable, check for a more
   restrictive organization or enterprise policy.
3. Run **Pre-commit auto-update** from the Actions tab on the default branch
   and confirm the job succeeds. When hook updates are available, confirm that
   a pull request is created or updated.

See [GitHub's Actions settings documentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository#preventing-github-actions-from-creating-or-approving-pull-requests).

## Documentation

Serve documentation locally:

```bash
mkdocs serve
```

## License

This project is licensed under the MIT License. See LICENSE for details.

---
