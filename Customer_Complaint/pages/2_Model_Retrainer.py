import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from back import BackgroundCSSGenerator

img1_path = r"C:\Users\jofra\Desktop\Mentorness\Background.jpeg"
img2_path = r"C:\Users\jofra\Desktop\Mentorness\Sidebar_Background.jpeg"
background_generator = BackgroundCSSGenerator(img1_path, img2_path)
page_bg_img = background_generator.generate_background_css()
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load the pre-trained vectorizer
vectorizer_path = r'C:\Users\jofra\Desktop\Mentorness\tfidf_vectorizer.pkl'
vectorizer = joblib.load(vectorizer_path)

# Function to adjust model based on average feedback score
def adjust_model(average_feedback_score, X_train_vec, X_test_vec, y_train, y_test):
    # Adjust model parameters based on feedback score
    if average_feedback_score < 1:
        model = LogisticRegression(max_iter=2000, random_state=42)
    elif 1 <= average_feedback_score < 2:
        model = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    elif 2 <= average_feedback_score < 3:
        model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    elif 3 <= average_feedback_score < 4:
        model = LogisticRegression(max_iter=1500, random_state=42)
    else:
        model = LogisticRegression(max_iter=1000, random_state=42)
    
    # Train the model
    model.fit(X_train_vec, y_train)
    
    # Save the adjusted model
    model_path = 'adjusted_customer_complaint_classifier.pkl'
    joblib.dump(model, model_path)
    
    return model

# Streamlit app for user feedback and model training
def main():
    st.title('Feedback for Model Adjustment')

    st.warning("We Will Be Assuming a Feedback Value Ranging From 1-5 For Now inorder to retrain the model but in actuality it is chosen from each Categories for every Narratives and Averaged But for Our convenience We will Assume that Final Feedback Value Between 1 and 5")
    
    # Allow user to upload preprocessed CSV file
    uploaded_file = st.file_uploader("Upload preprocessed CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)
        
        # Split data into features (X) and target (y)
        X = df['Narrative']
        y = df['product']
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Vectorize the narratives
        X_train_vec = vectorizer.transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Display slider for average feedback score
        average_feedback_score = st.slider('Average Feedback Score (1-5):', 1, 5, 3)
        st.write(f'Average Feedback Score: {average_feedback_score}')
        
        # Button to adjust model based on feedback
        if st.button('Adjust Model'):
            # Adjust the model based on the feedback score
            adjusted_model = adjust_model(average_feedback_score, X_train_vec, X_test_vec, y_train, y_test)
            st.success('Model adjusted and saved based on feedback!')

if __name__ == '__main__':
    main()
