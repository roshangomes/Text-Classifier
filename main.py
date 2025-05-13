# Use utils.py functions here
# Read the train.csv
import pandas as pd
#from data_inspect import read_Csv
import pytesseract
import cv2
import re
import os

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    denoised_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    denoised_img = cv2.cvtColor(denoised_img,cv2.COLOR_BGR2RGB)
    
    return denoised_img


def perform_ocr(image):
    # Set OCR engine and parameters
    pytesseract.pytesseract.tesseract_cmd = r"E:\tesseract\tesseract.exe"  # Replace with your Tesseract path
    custom_config = r'--oem 3 --psm 6'  # Adjust parameters as needed

    # Perform OCR
    text = pytesseract.image_to_string(image, config=custom_config)

    return text

def transform_short_forms_to_full(text):
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

def list_jpg_images(directory):
    # List to store the paths of jpg images
    jpg_images = []

    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file ends with .jpg or .JPG (case-insensitive)
            if file.lower().endswith(".jpg"):
                # Join the root directory with the file name to get the full path
                full_path = os.path.join(root, file)
                # Add a leading slash to the path (for Unix-like systems)
                jpg_images.append('/' + os.path.relpath(full_path, directory))
    
    return jpg_images

def process_images_and_update_df(images_folder, df, csvfile="./dataset/updated_train.csv"):
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
            if(i==100):
                break
            else:
                i+=1
            preprocessed_image = preprocess_image(image_path)
            
            # Perform OCR
            text = perform_ocr(preprocessed_image)
            
            # Transform short forms to full forms
            cleaned_text = transform_short_forms_to_full(text)
            
            # Append result to list
            ocr_texts.append(cleaned_text)
            
        else:
            ocr_texts.append("")  # If image does not exist, append empty string

    # Add OCR results to DataFrame
    df['text_output'] = ocr_texts

    # Save updated DataFrame to CSV
    df.to_csv(csvfile, index=False)


