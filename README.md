# Random Forest Prediction App

This project is a **Streamlit-based web application** designed to predict classes using a pre-trained Random Forest model. The app processes CSV files with 448 features, applies scaling and PCA transformations, and generates predictions.

---

## Features

- Upload a CSV file with 448 features for prediction.
- Automatically preprocesses data by:
    - Dropping non-feature columns (`hsi_id`, `vomitoxin_ppb`).
    - Scaling the data using a pre-trained scaler.
    - Reducing dimensionality using PCA.
- Predicts classes using a pre-trained Random Forest model.
- Displays predictions for all rows in the uploaded dataset.

---

## Installation

### Prerequisites

- Python 3.7 or higher
- Required libraries: `streamlit`, `joblib`, `numpy`, `pandas`


### Steps

1. Clone this repository:

```bash
git clone https://github.com/abhaystar2004/imagoai_ML_TASK;
cd imagoai_ML_TASK;
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure the following files are present in the project directory:
    - `rf_model.pkl` (Pre-trained Random Forest model)
    - `scaler.pkl` (Pre-trained scaler)
    - `pca.pkl` (Pre-trained PCA model)

---

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`.
3. Upload a CSV file containing **448 features** (excluding non-feature columns like `hsi_id` and `vomitoxin_ppb`).
4. View the predictions displayed in the app.

---

## Input File Format

The input CSV file should:

- Contain exactly **448 feature columns**.
- Exclude non-feature columns such as `hsi_id` and `vomitoxin_ppb`.

Example format:


| hsi_id | Feature_1 | Feature_2 | ... | Feature_448 | vomitoxin_ppb |
| :-- | :-- | :-- | :-- | :-- | :-- |
| imagoai_corn_0 | 0.416181 | 0.396844 | ... | 0.704520 | 1100 |

The app automatically drops the `hsi_id` and `vomitoxin_ppb` columns before processing.

---

## Project Structure

```
.
├── app.py                # Main Streamlit application
├── rf_model.pkl          # Pre-trained Random Forest model
├── scaler.pkl            # Pre-trained scaler
├── pca.pkl               # Pre-trained PCA model
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## Error Handling

The app includes error handling for:

- Invalid file uploads (e.g., non-CSV files).
- Incorrect number of features in the uploaded file.
- Internal processing errors with descriptive messages displayed in the UI.

---

## Example Output

After uploading a valid CSV file, the app will display predictions for each row:


| Row ID | Predicted Class |
| :-- | :-- |
| imagoai_corn_0 | Class A |
| imagoai_corn_1 | Class B |
| ... | ... |

---

## Future Improvements

- Add support for additional machine learning models.
- Enhance input validation with more detailed feedback.
- Provide visualizations of predictions and feature importance.

