#rom data_inspect import read_Csv
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os


# Define the entity-unit mapping
entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon', 'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
}

# Combine all the accepted units into one set for unit classification
all_units = set()
for unit_set in entity_unit_map.values():
    all_units.update(unit_set)

# Function to clean the text data
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text.lower()  # Convert to lowercase
    return ''

# Extract numeric value and unit from the 'entity_value'
def extract_numeric_and_unit(entity_value):
    tokens = entity_value.split()
    numeric_value = None
    unit_value = 'unknown'
    
    # Search through tokens to find a unit, then capture the preceding numeric value
    for i, token in enumerate(tokens):
        for unit in all_units:
            if unit in token.lower():
                unit_value = unit  # Capture the unit
                # Look for the numeric value before the unit
                if i > 0 and re.match(r'[\d.]+', tokens[i-1]):
                    numeric_value = tokens[i-1]  # Get the number before the unit
                return numeric_value, unit_value
    
    # Return 'unknown' if no unit or number is found
    return numeric_value or 'unknown', unit_value

def clean_df(df):
    # Apply the cleaning and extraction functions
    df['numeric_value'], df['unit'] = zip(*df['entity_value'].apply(extract_numeric_and_unit))

    # Prepare the data for classification (using unit as the target variable)
    X = df['text_output'].apply(clean_text)  # Cleaned text data as input
    y = df['unit']  # Units as the target variable

    # Split the dataset into training and test sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X,y


# Function to extract numeric value specifically in front of predicted class (unit)
def extract_numeric_in_front_of_unit(text_output, predicted_class):
    # Tokenize the text into words
    tokens = text_output.split()
    predicted_class_lower = predicted_class.lower()  # Normalize the predicted class for matching
    
    for i, token in enumerate(tokens):
        # Check if the token matches the exact predicted class (e.g., 'gram' is not in 'kilogram')
        if predicted_class_lower == token.lower():
            # Check if there is a numeric value right before the predicted class
            if i > 0 and re.match(r'[\d.]+', tokens[i-1]):
                return tokens[i-1]  # Return the numeric value found before the unit
    
    return 'unknown'  # Return 'unknown' if no numeric value is found


# Wrapper function to handle both numeric value and unit prediction
def predict_numeric_and_unit(text_output, predicted_class):
    numeric_value = extract_numeric_in_front_of_unit(text_output, predicted_class)
    
    # If numeric value is 'unknown', set unit to 'unknown' as well
    if numeric_value == 'unknown':
        predicted_class = 'unknown'
    
    return predicted_class, numeric_value


#svm model
def svm(csv_file):
    df=pd.read_csv(csv_file)
    x,y=clean_df(df)
    # Create a pipeline for TF-IDF and SVM
    preprocessor = ColumnTransformer(
    transformers=[
            ('entity_name_tfidf', TfidfVectorizer(), 'entity_name'),   # Process entity_name
            ('entity_value_tfidf', TfidfVectorizer(), 'entity_value')  # Process entity_value
        ]
    )
    X_train=pd.DataFrame({"entity_name":df["entity_name"],"entity_value":x})
    pipeline_svm = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', SVC(kernel='poly'))
    ])

    # Train the model
    pipeline_svm.fit(X_train, y)
    return pipeline_svm



def rf(csv_file):
    df=pd.read_csv(csv_file)
    x,y=clean_df(df)
    # Create a pipeline for TF-IDF and SVM
    preprocessor = ColumnTransformer(
    transformers=[
            ('entity_name_tfidf', TfidfVectorizer(), 'entity_name'),   # Process entity_name
            ('entity_value_tfidf', TfidfVectorizer(), 'entity_value')  # Process entity_value
        ]
    )
    X_train=pd.DataFrame({"entity_name":df["entity_name"],"entity_value":x})
    pipeline_rf = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier())
    ])

    # Train the model
    pipeline_rf.fit(X_train, y)
    return pipeline_rf
# Create a pipeline for TF-IDF and Random Forest

def essembler(csv_file, pipeline_svm, pipeline_rf):
    # Read the dataset
    df = pd.read_csv(csv_file)
    # Assuming clean_df preprocesses the DataFrame and returns X and y
    x, y = clean_df(df)
    X_train=pd.DataFrame({"entity_name":df["entity_name"],"entity_value":x})
    # Define the Voting Classifier using hard voting
    voting_clf = VotingClassifier(
        estimators=[
            ('svm', pipeline_svm),  # SVM pipeline
            ('rf', pipeline_rf)     # Random Forest pipeline
        ],
        voting='hard'
    )

    # Train the Voting Classifier
    voting_clf.fit(X_train, y)

    return voting_clf


def report(model,x,y):
    # Predict and evaluate
    y_pred_voting = model.predict(x)
    print(classification_report(y, y_pred_voting))
    
def graph(y_pred,y,filename):

    # Create the results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_pred, y)
    # Get class names from the unique values in t2["prediction"]
    class_names = np.unique(y.unique())
    #   Create and save the confusion matrix plot
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix {filename}')
    plt.savefig(f"results/confusion_matrix{filename}.png")
    plt.show()
    plt.close()