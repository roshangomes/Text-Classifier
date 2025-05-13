import os
import random
import sys
import pandas as pd
from data_inspect import read_Csv,download
from main import process_images_and_update_df
from model import clean_df, clean_text,svm,rf,essembler,report,graph,predict_numeric_and_unit
import numpy as np
from sanity import sanity_check

import re
global svc
global rfm
global esm

def train_call(file="./dataset/train.csv"):
    df=read_Csv(file)
    print(df.head())
    df.to_csv('./dataset/TrainDataset.csv', index=False)
    return df

def read_and_process_train(filename='./train_images3'):
    df=pd.read_csv("./dataset/TrainDataset.csv")
    df=df.iloc[:100,:]
    process_images_and_update_df(filename,df)
    
def train_func_exe():
    df=train_call()
    download(df)
    
def main_func_exe():
    read_and_process_train()
  
def model_func_exe():
    global svc,esm,rfm
    df=pd.read_csv("./dataset/updated_train.csv")
    x,y=clean_df(df)
    X_train=pd.DataFrame({"entity_name":df["entity_name"],"entity_value":x})
    
    svc=svm("./dataset/updated_train.csv")
    pred=svc.predict(X_train)
    report(svc,X_train,y)
    graph(pred,y,'svm')
    
    
    
    rfm=rf("./dataset/updated_train.csv")
    x,y=clean_df(df)
    X_train=pd.DataFrame({"entity_name":df["entity_name"],"entity_value":x})
    
    pred=rfm.predict(X_train)
    report(rfm,X_train,y)
    graph(pred,y,'rf') 
    
    esm=essembler("./dataset/updated_train.csv",svc,rfm)
    x,y=clean_df(df)
    X_train=pd.DataFrame({"entity_name":df["entity_name"],"entity_value":x})
    
    pred=esm.predict(X_train)
    report(esm,X_train,y)
    graph(pred,y,'esm') 
    
    return svc,rfm,esm  

    
def test(filename1="./dataset/sample_test.csv",filename2="./dataset/sample_test_out.csv"): 
    t1=pd.read_csv(filename1) 
    #t1=t1.fillna(" ")
    download(t1,"test_images")
    
    process_images_and_update_df_test("test_images",t1)
    t2=pd.read_csv(filename2)
    t1=t1.fillna(" ")
    # Prepare the data for classification (using unit as the target variable)
    Xt = t1['text_output'].apply(clean_text)  # Cleaned text data as input
    l2=t2["prediction"].to_list() # Units as the target variable
    X_train=pd.DataFrame({"entity_name":t1["entity_name"],"entity_value":Xt})
    svc=svm("./dataset/updated_train.csv")
    rfm=rf("./dataset/updated_train.csv")
    esm=essembler("./dataset/updated_train.csv",svc,rfm)
    
    pred=esm.predict(X_train)
    
    temp_df=pd.read_csv("./dataset/updated_train.csv")
    temp_x,temp_y=clean_df(temp_df)
    X_train_temp=pd.DataFrame({"entity_name":temp_df["entity_name"],"entity_value":temp_x})
    temp_pred=esm.predict(X_train_temp)
    
    l2 = [i if not pd.isna(i) else '1 gram' for i in l2]
    l2=[str(i).split()[1] for i in l2]
    report(esm,X_train_temp,temp_y)
    graph(temp_pred,temp_y,'esm test') 
    
    l=t2["prediction"].to_list()
    
    # Initialize an empty list to store concatenated results
    predicted_results = []

    for i in range(len(t1)):
        text_output = t1.iloc[i]["text_output"]  # Access the 'entity_value' for the current row
        predicted_class = pred[i]  # Get the predicted class for the current row
        predicted_class, numeric_value = predict_numeric_and_unit(text_output, predicted_class)  # Get the numeric and class
    
        # Concatenate the predicted class and numeric value
        concatenated_result = f"{numeric_value} {predicted_class}"
    
        # Append the concatenated result to the list
        predicted_results.append(concatenated_result)

    # Now 'predicted_results' will contain concatenated strings like "100 gram" or "unknown unknown"
    predicted_results = [i if i != 'unknown unknown' else "" for i in predicted_results]
    #predicted_results

    output=pd.DataFrame({"prediction":predicted_results})
    output.to_csv('Output.csv',index=False)
    # Reset index to include index as a column
    df_reset = output.reset_index()
    #Rename the index column to 'index'
    df_reset.rename(columns={'index': 'index'}, inplace=True)
    # Save to CSV
    df_reset.to_csv('Output.csv', index=False)  #
    
    # Add the path to 'src' where 'utils.py' is located
    sys.path.append(os.path.abspath('./src'))
    # Import the utils.py file
    sanity_check("Output.csv","./dataset/sample_test_out.csv")
    return predicted_results
    
    
def process_images_and_update_df_test(images_folder, df):
    # Initialize a list to store OCR results
    ocr_texts = []
    i=0
    for index, row in df.iterrows():
        image_url = row['image_link']
        image_name = os.path.basename(image_url)
        image_path = os.path.join(images_folder, image_name)
        #image_path = "../"+images_folder+"/"+image_name
        
        if os.path.exists(image_path):
            # Preprocess the image
            print(image_path, image_name,i)
            i+=1
            preprocessed_image = preprocess_image_test(image_path)
            
            # Perform OCR
            text = perform_ocr_test(preprocessed_image)
            
            # Transform short forms to full forms
            cleaned_text = transform_short_forms_to_full_test(text)
            
            # Append result to list
            ocr_texts.append(cleaned_text)
            
        else:
            ocr_texts.append("")  # If image does not exist, append empty string

    # Add OCR results to DataFrame
    df['text_output'] = ocr_texts

    # Save updated DataFrame to CSV
    df.to_csv('./dataset/updated_test.csv', index=False)

import pytesseract

def perform_ocr_test(image):
    # Set OCR engine and parameters
    pytesseract.pytesseract.tesseract_cmd = r"E:\tesseract\tesseract.exe"  # Replace with your Tesseract path
    custom_config = r'--oem 3 --psm 6'  # Adjust parameters as needed

    # Perform OCR
    text = pytesseract.image_to_string(image, config=custom_config)

    return text

import cv2
def preprocess_image_test(image_path):
    img = cv2.imread(image_path)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    denoised_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    #denoised_img = cv2.cvtColor(denoised_img,cv2.COLOR_BGR2RGB)
    
    return denoised_img

    
def transform_short_forms_to_full_test(text):
    # Mapping of common short forms to acceptable full forms, including multi-word units without spaces
    unit_short_to_full = {
        'cm': 'centimetre', 'mm': 'millimetre', 'm': 'metre', 'ft': 'foot',
        'in': 'inch', 'yd': 'yard', 'g': 'gram', 'kg': 'kilogram',
        'mg': 'milligram', 'mcg': 'microgram', 'oz': 'ounce', 'lb': 'pound',
        't': 'ton', 'kv': 'kilovolt', 'v': 'volt', 'mv': 'millivolt',
        'w': 'watt', 'kw': 'kilowatt', 'cl': 'centilitre', 'ml': 'millilitre',
        'l': 'litre', 'fl oz': 'fluid ounce', 'gal': 'gallon', 'qt': 'quart',
        'pt': 'pint', 'cu ft': 'cubic foot', 'cu in': 'cubic inch',
        'cuft': 'cubic foot', 'cuin': 'cubic inch', 'floz': 'fluid ounce'
    }

    # Regular expression to match numeric values followed by short forms, with or without space
    pattern = re.compile(r'(\d+)\s*(' + '|'.join(re.escape(key) for key in unit_short_to_full.keys()) + r')\b', re.IGNORECASE)

    # Function to replace short forms with their full forms
    def replace_match(match):
        unit = match.group(2).lower()  # Convert the unit to lowercase for consistency
        if unit in unit_short_to_full:
            return f"{match.group(1)} {unit_short_to_full[unit]}"
        return match.group(0)  # Return the original match if no replacement is found

    # Perform the substitution in the text
    return pattern.sub(replace_match, text)    
    
def predictor():
    '''
    Call your model/approach here
    '''
    train_func_exe()
    main_func_exe()
    model_func_exe()
    
    #test_df=pd.read_csv("./dataset/test.csv")
    #test_df=test_df.iloc[:20,:]
    #test_df.to_csv('./dataset/test20.csv', index=False)
    results=test()
    
    
    #svc=svm("./dataset/updated_train.csv")
    #rfm=rf("./dataset/updated_train.csv")
    #esm=essembler("./dataset/updated_train.csv",svc,rfm)
    
    #pred=esm.predict([[]])
    
    return results

if __name__ == "__main__":
    
    #print("Train Func Exe")
    #train_func_exe()
    #print("Main Func Exe")
    #main_func_exe()
    #print("Model Func Exe")
    #model_func_exe()
    
    #DATASET_FOLDER = './dataset/'
      #
    
    #test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    #test['prediction'] = test.apply(
    #    lambda row: predictor(row['image_link'], row['group_id'], row['entity_name']), axis=1)
    
    #output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    #test[['index', 'prediction']].to_csv(output_filename, index=False)
    predictor()