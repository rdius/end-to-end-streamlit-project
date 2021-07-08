#############"
# Building App
#############
# Core pkgs
import streamlit as st
import altair as alt
## EDA Pkgs
import pandas as pd
import numpy as np
## Utils
import joblib

## Load Emotion model
pipe_lr = joblib.load('../models/emotion_classification_pipe_lr06.pkl')

## Fxn
def predict_emotion(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results
    
    
## Emojis
emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}


def main():
    st.title('Emotion Classifier App')
    menu = ["Home", "Monitor", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        st.subheader("Home-Emotion In Text")
        with st.form(key = 'emotion_clf_form'):
            raw_text = st.text_area("Type your text here")
            submit_text = st.form_submit_button(label = 'submit')
        if submit_text:
            col1,col2 = st.beta_columns(2)
            ## Apply fxn
            prediction = predict_emotion(raw_text)
            probability = get_prediction_proba(raw_text)
            with col1:
                st.success('Original Text')
                st.write(raw_text)
                
                st.success("Prediction")
                emoji_icon = emotions_emoji_dict
                st.write(prediction)
                #st.write("{}:{}".format(prediction), emoji_icon)
                st.write("confidence:{}".format(np.max(probability)))
                
                
            with col2:
                st.success("Prediction Probability")
                #st.write(probability)
                proba_df = pd.DataFrame(probability, columns = pipe_lr.classes_)
                #st.write(proba_df.T)
                proba_clean = proba_df.T.reset_index()
                proba_clean.columns = ['emotion', 'probability']
                
                fig = alt.Chart(proba_clean).mark_bar().encode(x ='emotion', y ='probability')
                st.altair_chart(fig, use_container_width = True)
                
                
    elif choice == "Monitor":
        st.subheader("Monitor App")
    
    else :
        st.subheader("About")
        
        
        
if __name__ == '__main__':
    main()