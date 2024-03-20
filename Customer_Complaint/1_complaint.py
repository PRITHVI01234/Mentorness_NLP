import os
import streamlit as st
import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import nltk
from back import BackgroundCSSGenerator

img1_path = r"C:\Users\jofra\Desktop\Mentorness\NLP Intern\Customer_Complaint\Background.jpeg"
img2_path = r"C:\Users\jofra\Desktop\Mentorness\NLP Intern\Customer_Complaint\Sidebar_Background.jpeg"
background_generator = BackgroundCSSGenerator(img1_path, img2_path)
page_bg_img = background_generator.generate_background_css()
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function for text preprocessing
def preprocess(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Function to classify complaint narratives
def classify_complaints(narratives, vectorizer, model):
    # Preprocess the narratives
    preprocessed_narratives = [preprocess(narrative) for narrative in narratives]
    
    # Vectorize the preprocessed narratives
    X = vectorizer.transform(preprocessed_narratives)
    
    # Classify the complaints using the trained model
    categories = model.predict(X)
    
    return categories

# Function to generate separate CSV files for each category and create Python files
def generate_files(narratives, categories):
    category_list = list(set(categories))
    for category in category_list:
        # Filter narratives by category
        category_narratives = [narratives[i] for i, cat in enumerate(categories) if cat == category]
        
        # Create a DataFrame for the category
        df = pd.DataFrame({'Narrative': category_narratives})
        
        # Save the DataFrame to a CSV file
        csv_filename = f'{category}_complaints.csv'
        
        # Remove existing CSV file if it exists
        if os.path.exists(csv_filename):
            os.remove(csv_filename)
        
        df.to_csv(csv_filename, index=False)
        
        # Create Python file for feedback and average calculation
        python_code = f"""
import streamlit as st
import pandas as pd
import numpy as np
from back import BackgroundCSSGenerator

img1_path = "C:/Users/jofra/Desktop/Mentorness/Background.jpeg"
img2_path = "C:/Users/jofra/Desktop/Mentorness/Sidebar_Background.jpeg"
background_generator = BackgroundCSSGenerator(img1_path, img2_path)
page_bg_img = background_generator.generate_background_css()
st.markdown(page_bg_img, unsafe_allow_html=True)

def main():
    st.title('{category} Complaints')
    
    # Load the complaints data
    df = pd.read_csv('{csv_filename}')
    
    # Display the complaints data
    st.write(df)
    st.warning("You Can Assign Feedback Score Ranging From 1 - 5 for each of the Narrative But We Will Not do that since it is Time Consuming")
    st.warning("We will Assume a value for Avg FeedBack in Each Case")
        
if __name__ == '__main__':
    main()

"""
        # Write Python code to file
        with open(os.path.join('pages', f'{category}_page.py'), 'w') as file:
            file.write(python_code)
    
# Streamlit app
def main():
    st.title('Bank Complaint Classifier')
    
    
    # Allow user to upload a CSV file or type narratives into a text area
    upload_file = st.file_uploader('Upload CSV file with complaint narratives:', type=['csv'])
    if upload_file is not None:
        # Read the uploaded CSV file
        df = pd.read_csv(upload_file)
        narratives = df['Narrative'].tolist()
    
        # Allow user to upload the model file
        st.warning("You Can Choose the Improvised Model which was created based on Feedback")
        model_file = st.file_uploader('Upload model file:', type=['pkl'])
        if model_file is not None:
            model = joblib.load(model_file)
            vectorizer_path = 'tfidf_vectorizer.pkl'  # Assuming the vectorizer path remains constant
            vectorizer = joblib.load(vectorizer_path)
            
            if st.button('Classify Complaints'):
                # Classify the complaint narratives
                categories = classify_complaints(narratives, vectorizer, model)
                
                # Generate separate CSV files for each category and create Python files
                generate_files(narratives, categories)
                
                st.success('Complaints classified, CSV files generated, and Python files created successfully!')

if __name__ == '__main__':
    main()
