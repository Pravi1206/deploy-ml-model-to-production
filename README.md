Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, WSL1 or 2 is recommended.

# Environment Set up
**Using pip and venv (Recommended)**
    * Ensure you have Python 3.8+ installed
    * Create virtual environment: `python3 -m venv .venv`
    * Activate environment: `source .venv/bin/activate` (On Windows: `.venv\Scripts\activate`)
    * Install dependencies:
      ```bash
      pip install -r starter/requirements.txt
      pip install -r starter/requirements-dev.txt  # For testing and development
      ```

## Continuous Integration

The repository uses GitHub Actions for continuous integration. The workflow is configured in `.github/workflows/ci.yml` and automatically runs on every push to the `main` or `master` branch. The CI pipeline:

1. Sets up Python 3.10 environment
2. Installs all dependencies from `requirements.txt` and `requirements-dev.txt`
3. Runs `flake8` for code quality checks (syntax errors and undefined names fail the build)
4. Runs `pytest` to ensure all tests pass

**All tests must pass and flake8 must complete without errors for the build to succeed.** You can view the status of CI runs in the "Actions" tab of the GitHub repository.

## Training the Model

To train the model, run the following command from the project root:

```bash
# Activate your virtual environment first
source .venv/bin/activate  # or activate your conda environment

# Run the training script
python starter/train_model.py
```

The training script will:
1. Load and clean the census data (creates `census_clean.csv`)
2. Split data into train/test sets (80/20 split)
3. Process features (one-hot encode categorical, keep continuous)
4. Train a Random Forest Classifier
5. Evaluate on test set and print metrics (Precision, Recall, F-beta)
6. Compute performance on data slices for each categorical feature
7. Save model artifacts to `model/`:
   - `model.pkl` - Trained Random Forest model
   - `encoder.pkl` - OneHotEncoder for categorical features
   - `lb.pkl` - LabelBinarizer for target labels
   - `slice_output.txt` - Performance metrics on data slices

## Testing the Model

Run all tests using pytest from the project root:

```bash
# Run all tests
pytest -v

# Run specific test files
pytest starter/ml/test_model.py -v        # Model unit tests
pytest api/test_router.py -v              # API unit tests

# Run with coverage report
pytest --cov=starter --cov-report=html
```

**Model Tests** (`test_model.py`):
- `test_train_model()` - Verifies model training and hyperparameters
- `test_compute_model_metrics()` - Tests metric calculations with perfect predictions
- `test_inference()` - Tests prediction functionality and output format
- `test_model_metrics_with_partial_accuracy()` - Tests metrics with partial accuracy

**API Tests** (`test_router.py`):
- `test_get_root()` - Tests GET endpoint returns welcome message
- `test_post_predict_below_50k()` - Tests POST with data likely earning <=50K
- `test_post_predict_above_50k()` - Tests POST with data likely earning >50K
- `test_post_predict_malformed_data()` - Tests error handling with malformed data

## Running the API

To run the FastAPI application locally using uvicorn:

```bash
# From the starter directory
cd starter
uvicorn main:app --host 0.0.0.0 --port 8000

# Or with auto-reload for development (automatically reloads on code changes)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Or from the project root
cd /home/fixcfhu/repos/Pravi1206/deploy-ml-model-to-production
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will start on `http://localhost:8000`. You can access:

- **Root endpoint**: `http://localhost:8000/` - Welcome message
- **Interactive API docs**: `http://localhost:8000/docs` - Swagger UI with interactive API documentation
- **Alternative docs**: `http://localhost:8000/redoc` - ReDoc documentation
- **Prediction endpoint**: `POST http://localhost:8000/predict` - Make income predictions

The `/docs` endpoint provides an interactive interface where you can:
- View all available endpoints
- See request/response schemas
- Test the API directly in your browser
- View example requests and responses

### Testing the API with a Live POST Request

To test the API with a live POST request using the requests module:

```bash
# Make sure the API is running first (in another terminal)
# Then run the live POST script
python api/live_post.py
```

This script will send two POST requests to the API and display:
- Input data (formatted JSON)
- HTTP status code
- Prediction result (<=50K or >50K)
