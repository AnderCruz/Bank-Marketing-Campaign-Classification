# Bank Marketing Investment Prediction

This project analyzes data from a bank's marketing campaign to predict whether a client will subscribe to a term deposit investment. Using machine learning, this notebook walks through the entire process from data exploration and preprocessing to model training, evaluation, and selection.

## Project Goal

The primary objective is to build a classification model that can accurately predict if a bank client will subscribe to a term deposit based on data from marketing campaigns. This helps the bank to better target its marketing efforts and understand the key factors that influence a client's decision.

## Dataset

The dataset used is `investment_marketing.csv`, which contains information about clients contacted during the campaign.

**Features:**

| Original Name           | English Name              | Description                                      | Data Type |
|-------------------------|---------------------------|--------------------------------------------------|-----------|
| `idade`                 | `age`                     | Client's age.                                    | Numerical |
| `estado_civil`          | `marital_status`          | Marital status ('casado (a)', 'solteiro (a)', 'divorciado (a)'). | Categorical|
| `escolaridade`          | `education`               | Education level ('superior', 'medio', 'fundamental'). | Categorical|
| `inadimplencia`         | `default`                 | Has credit in default? ('sim', 'nao').           | Categorical|
| `saldo`                 | `balance`                 | Average yearly balance.                          | Numerical |
| `fez_emprestimo`        | `loan`                    | Has a personal loan? ('sim', 'nao').             | Categorical|
| `tempo_ult_contato`     | `last_contact_duration`   | Duration of the last contact in seconds.         | Numerical |
| `numero_contatos`       | `number_of_contacts`      | Number of contacts performed during this campaign. | Numerical |
| **`aderencia_investimento`** | **`investment_subscribed`** | **(Target)** Has the client subscribed to the investment? ('sim', 'nao'). | Categorical|

## Workflow

The project follows a standard machine learning pipeline:

1.  **Data Loading & Initial Analysis**: The dataset is loaded using Pandas. An initial check for null values and data types is performed.
2.  **Exploratory Data Analysis (EDA)**: The data is visualized using Plotly to identify patterns, understand distributions, and see how different features relate to the target variable (`investment_subscribed`).
3.  **Data Preprocessing & Transformation**:
      * Categorical features are converted into a numerical format using `OneHotEncoder`.
      * The binary target variable is converted into 0s and 1s using `LabelEncoder`.
      * The data is split into training and testing sets, stratified by the target variable to ensure balanced representation.
4.  **Model Training & Comparison**: Three classification models are trained and evaluated:
      * **Dummy Classifier**: Serves as a baseline for performance comparison.
      * **Decision Tree Classifier**: A simple yet powerful model. The tree's depth is limited (`max_depth=3`) to prevent overfitting.
      * **K-Nearest Neighbors (KNN)**: A distance-based algorithm. The data for this model is first normalized using `MinMaxScaler`.
5.  **Model Selection & Saving**: The models' accuracies are compared, and the best-performing model is selected. The final model and its associated data transformer are saved as `.pkl` files for future use.

## Exploratory Data Analysis (EDA)

The EDA provided several key insights:

  - The dataset is imbalanced, with significantly more clients not subscribing to the investment than those who did.
  - **Last Contact Duration (`tempo_ult_contato`)** appears to be a strong predictor. The box plot shows that clients who subscribed to the investment generally had a much longer last contact duration.
  - Clients with a higher education level (`superior`) showed a proportionally higher rate of subscription compared to other levels.
  - Other variables like `age`, `balance`, and `marital_status` also showed some influence on the subscription decision.

## Modeling and Evaluation

The data was split into a training set (75%) and a testing set (25%). The following models were trained and their accuracy was measured on the test set.

  - **Baseline Model**: A `DummyClassifier` that always predicts the most frequent class ('no') was used to set a baseline accuracy.
  - **Decision Tree**: This model was chosen for its high explainability. To avoid overfitting (where the model learns the training data too well but fails on new data), its `max_depth` was tuned to 3. This resulted in a model that generalizes better to unseen data.
  - **K-Nearest Neighbors (KNN)**: As a distance-based algorithm, KNN requires features to be on a similar scale. Therefore, the data was normalized using `MinMaxScaler` before training.

## Results

The final accuracy scores for each model on the test data were as follows:

| Model                    | Test Accuracy |
|--------------------------|---------------|
| Dummy Classifier         | 60.25%        |
| K-Nearest Neighbors (KNN)| 68.77%        |
| **Decision Tree (max\_depth=3)** | **71.61%** |

The **Decision Tree Classifier** achieved the highest accuracy on the test set and was selected as the final model. It provides a significant improvement over the baseline and performs better than the KNN model in this scenario.

## Technologies Used

  - **Python 3**
  - **Pandas**: For data manipulation and analysis.
  - **Scikit-learn**: For data preprocessing, model training and evaluation.
  - **Plotly**: For creating interactive data visualizations.
  - **Matplotlib**: For visualizing the decision tree.
  - **Pickle**: For saving and loading the trained model.

## How to Use the Model

The best model (`DecisionTreeClassifier`) and its `OneHotEncoder` have been saved to `modelo_arvore.pkl` and `modelo_onehotenc.pkl`, respectively. You can use these files to make predictions on new data.

Here is an example of how to load the models and predict a new client's subscription likelihood:

```python
import pickle
import pandas as pd

# Load the saved models
with open('modelo_onehotenc.pkl', 'rb') as f:
    one_hot_encoder = pickle.load(f)

with open('modelo_arvore.pkl', 'rb') as f:
    decision_tree_model = pickle.load(f)

# Create a new data point for prediction
new_client_data = {
    'idade': [45],
    'estado_civil': ['solteiro (a)'],
    'escolaridade': ['superior'],
    'inadimplencia': ['nao'],
    'saldo': [23040],
    'fez_emprestimo': ['nao'],
    'tempo_ult_contato': [800],
    'numero_contatos': [4]
}
new_client_df = pd.DataFrame(new_client_data)

# Apply the same OneHotEncoding transformation
new_client_transformed = one_hot_encoder.transform(new_client_df)

# Make a prediction
prediction = decision_tree_model.predict(new_client_transformed)

# Interpret the result
if prediction[0] == 1:
    print("Prediction: The client WILL subscribe to the investment.")
else:
    print("Prediction: The client WILL NOT subscribe to the investment.")

# Expected Output: Prediction: The client WILL subscribe to the investment.
```

## How to Run This Project

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**

    ```bash
    pip install pandas scikit-learn plotly matplotlib
    ```

4.  **Run the Jupyter Notebook:**

    ```bash
    jupyter notebook Project_Investment_Marketing.ipynb
    ```
