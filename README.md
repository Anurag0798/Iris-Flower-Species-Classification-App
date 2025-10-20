# Iris Flower Species Classification App

## Overview
The Iris Flower Species Classification App is a Streamlit application that accepts sepal and petal measurements, uses classification models, predicts the Iris species (setosa, versicolor, virginica), and can optionally store inputs and predictions for analysis. The repository contains the Streamlit app, a requirements file, and a model artifact (pickle/joblib).

Repository:
```
git clone https://github.com/Anurag0798/Iris-Flower-Species-Classification-App.git
```

Note: If the repository contains example connection strings or credentials (e.g., for optional telemetry), do NOT commit real credentials to source control - use environment variables or a secrets manager instead.

## Features
* Interactive Streamlit UI for entering sepal/petal measurements and getting an immediate species prediction.
* Loads a pre-trained classification model (pickle/joblib) to produce predictions.
* Displays predicted class and, if available, class probabilities.
* Optional hooks can be added to persist inputs + predictions to a file or database for telemetry/analysis.

## Technologies Used
* Python
* Streamlit
* scikit-learn (model)
* pandas
* NumPy
* pickle or joblib (for model serialization)

## Requirements
Install the packages listed in `requirements.txt`. If the included requirements file is minimal, add any missing packages (streamlit, pandas, numpy, scikit-learn, joblib/pickle helper).

Example:
```
pip install -r requirements.txt
```

Recommended dependencies to include:
* streamlit
* pandas
* numpy
* scikit-learn
* joblib (or use pickle)

## Inputs (UI fields)
The Streamlit app collects the following numeric inputs (implemented as `number_input` fields):
* sepal length (cm)
* sepal width (cm)
* petal length (cm)
* petal width (cm)

Do not change the input order or scaling without updating the preprocessing and model accordingly.

## How to Run
1. Clone the repository:
```
git clone https://github.com/Anurag0798/Iris-Flower-Species-Classification-App.git
```
```
cd Iris-Flower-Species-Classification-App
```

2. Create and activate a virtual environment (recommended):
```
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows
```

3. Install dependencies:
```
pip install -r requirements.txt
```
(If `requirements.txt` is missing packages, also run: `pip install streamlit pandas numpy scikit-learn joblib`)

4. Place the model file:
* Ensure the model artifact is in the project directory, or update the model-loading path in the app.

5. Run the Streamlit app:
```
streamlit run iris_app.py
```

## Usage
* Launch the app in your browser via Streamlit.
* Enter the sepal and petal measurements listed above.
* Click the predict button to obtain the predicted Iris species.
* The inputs and outputs will also be persisted in the MongoDB database.

## Troubleshooting
* If model loading fails, confirm the model file exists and was serialized with a compatible Python and scikit-learn version.
* If predictions look incorrect, verify that preprocessing (scaling, encoding) applied at inference matches training.
* If Streamlit fails to start, ensure Streamlit is installed and no port conflicts exist.
* If any required package is missing at runtime, add it to `requirements.txt` and reinstall dependencies.

## Contributing
Contributions are welcome. Suggested workflow:
* Fork the repo.
* Create a feature branch.
* Implement changes and update README and requirements.
* Open a pull request describing the changes.

Please avoid adding secrets or large model files containing sensitive information to commits.

## License
This project is licensed under the MIT License - see the LICENSE file for details.