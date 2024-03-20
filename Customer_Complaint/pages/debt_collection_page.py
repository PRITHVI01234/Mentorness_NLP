
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
    st.title('debt_collection Complaints')
    
    # Load the complaints data
    df = pd.read_csv('debt_collection_complaints.csv')
    
    # Display the complaints data
    st.write(df)
    st.warning("You Can Assign Feedback Score Ranging From 1 - 5 for each of the Narrative But We Will Not do that since it is Time Consuming")
    st.warning("We will Assume a value for Avg FeedBack in Each Case")
        
if __name__ == '__main__':
    main()

