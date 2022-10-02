# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 06:22:08 2022

@author: Ranjith Kumar Raja R
"""

import pickle
import pandas as pd
import altair as alt
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.utils.validation import check_array
import numpy as np

diabetes_model = pickle.load(open('C:/Users/Ranjith Kumar Raja R/OneDrive - University of Hertfordshire/Desktop/Multiple disease prediction/saved/diabetes_model.sav','rb'))
heart_disease_model = pickle.load(open('C:/Users/Ranjith Kumar Raja R/OneDrive - University of Hertfordshire/Desktop/Multiple disease prediction/saved/heart_disease_model.sav','rb'))
parkinsons_model = pickle.load(open('C:/Users/Ranjith Kumar Raja R/OneDrive - University of Hertfordshire/Desktop/Multiple disease prediction/saved/parkinsons_model.sav','rb'))
medical_insurance = pickle.load(open('C:/Users/Ranjith Kumar Raja R/OneDrive - University of Hertfordshire/Desktop/Multiple disease prediction/saved/medical_insurance1.sav','rb'))
Breast_cancer = pickle.load(open('C:/Users/Ranjith Kumar Raja R/OneDrive - University of Hertfordshire/Desktop/Multiple disease prediction/saved/cancer.sav','rb'))
kidney_disease = pickle.load(open('C:/Users/Ranjith Kumar Raja R/OneDrive - University of Hertfordshire/Desktop/Multiple disease prediction/saved/kidney_disease.sav','rb'))


# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Multiple Disease Prediction System',
                          
                          ['Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Parkinsons Prediction',
                           'Medical insurance',
                           'Breast cancer',
                           'DNA',
                           'Kidney_Disease'],
                          icons=['activity','heart','person'],
                          default_index=0)
    
if(selected == 'DNA'):
    
    
    st.write("""
             # DNA Nucleotide Count Web App
             This app counts the nucleotide composition of query DNA!
             ***
             """)
             
    st.header('Enter DNA sequence')
    sequence_input = ">DNA Query 2\nGAACACGTGGAGGCAAACAGGAAGGTGAAGAAGAACTTATCCTATCAGGACGGAAGGTCCTGTGCTCGGG\nATCTTCCAGACGTCGCGACTCTAAATTGCCCCCTCTGAGGTCAAGGAACACAAGATGGTTTTGGAAATGC\nTGAACCCGATACATTATAACATCACCAGCATCGTGCCTGAAGCCATGCCTGCTGCCACCATGCCAGTCCT"
    sequence = st.text_area("Sequence input", sequence_input, height=250)
    sequence = sequence.splitlines()
    sequence = sequence[1:] # Skips the sequence name (first line)
    sequence = ''.join(sequence) # Concatenates list to string

    st.write("""
    ***
    """)

    ## Prints the input DNA sequence
    st.header('INPUT (DNA Query)')
    sequence

    ## DNA nucleotide count
    st.header('OUTPUT (DNA Nucleotide Count)')

    ### 1. Print dictionary
    st.subheader('1. Print dictionary')
    st.subheader('1. Print dictionary')
    def DNA_nucleotide_count(seq):
      d = dict([
                ('A',seq.count('A')),
                ('T',seq.count('T')),
                ('G',seq.count('G')),
                ('C',seq.count('C'))
                ])
      return d

    X = DNA_nucleotide_count(sequence)

    #X_label = list(X)
    #X_values = list(X.values())

    X

    ### 2. Print text
    st.subheader('2. Print text')
    st.write('There are  ' + str(X['A']) + ' adenine (A)')
    st.write('There are  ' + str(X['T']) + ' thymine (T)')
    st.write('There are  ' + str(X['G']) + ' guanine (G)')
    st.write('There are  ' + str(X['C']) + ' cytosine (C)')

    ### 3. Display DataFrame
    st.subheader('3. Display DataFrame')
    df = pd.DataFrame.from_dict(X, orient='index')
    df = df.rename({0: 'count'}, axis='columns')
    df.reset_index(inplace=True)
    df = df.rename(columns = {'index':'nucleotide'})
    st.write(df)

    ### 4. Display Bar Chart using Altair
    st.subheader('4. Display Bar chart')
    p = alt.Chart(df).mark_bar().encode(
        x='nucleotide',
        y='count'
    )
    p = p.properties(
        width=alt.Step(80)  # controls width of bar.
    )
    st.write(p)

# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    
    # page title
    st.title('Diabetes Prediction using ML')
    
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    
    with col2:
        Insulin = st.text_input('Insulin Level')
    
    with col3:
        BMI = st.text_input('BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diab_prediction[0] == 1):
          diab_diagnosis = 'The person is diabetic'
        else:
          diab_diagnosis = 'The person is not diabetic'
        
    st.success(diab_diagnosis)
    



# Diabetes Prediction Page
if (selected == 'Kidney_Disease'):
    
    # page title
    st.title('Kidney disease using ML')
    
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('age')
        
    with col2:
        bp = st.text_input('bp')
    
    with col3:
        al = st.text_input('al')
    
    with col1:
        pcc = st.text_input('pcc')
    
    with col2:
        bgr = st.text_input('bgr')
    
    with col3:
        bu = st.text_input('bu')
    
    with col1:
        sc = st.text_input('sc')
    
    with col2:
        hemo = st.text_input('hemo')
        
    with col3:
        pcv = st.text_input('pcv')
        
    with col1:
         htn = st.text_input('htn')
         
    with col3:
         dm = st.text_input('dm')
         
    with col1:
         appet = st.text_input('appet')

        
    
    
    # code for Prediction
    kidney_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Test Result'):
        kidney_prediction = kidney_disease.predict([[age,bp,al,pcc,bgr,bu,sc,hemo,pcv,htn,dm,appet]])
        
        if (kidney_prediction[0] == 1):
          kidney_diagnosis = 'The person is diabetic'
        else:
          kidney_diagnosis = 'The person is not diabetic'
        
    st.success(kidney_diagnosis)








# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Sex')
        
    with col3:
        cp = st.text_input('Chest Pain types')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
     
     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)
        
    
    

# Parkinson's Prediction Page
if (selected == "Parkinsons Prediction"):
    
    # page title
    st.title("Parkinson's Disease Prediction using ML")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
          parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)




    
    

if (selected == 'Medical insurance'):
    st.title('Medical Insurance using ML')
    
    
    col1, col2, col3 =st.columns(3)
    
    
    
    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex')
    with col3:
        bmi = st.text_input('BMI')
    with col1:
        children = st.text_input('Children')
    with col2:
        smoker = st.text_input('smoker')
    with col3:
        region = st.text_input('Region')
    
    
    #creating a button for prediction 
    if st.button('Insurance Result'):
        insurance_prediction = medical_insurance.predict([[age, sex, bmi, children, smoker, region]])
        x=insurance_prediction[0]
            
        st.success(x)

  
    

if (selected == 'Breast cancer'):
    st.title('Medical Insurance using ML')
    
    
    col1, col2, col3 =st.columns(3)
    
    
    
    with col1:
        radius_mean = st.text_input('radius_mean')
    with col2:
        texture_mean = st.text_input('texture_mean')
    with col3:
        perimeter_mean = st.text_input('perimeter_mean')
    with col1:
        area_mean = st.text_input('area_mean')
    with col2:
        smoothness_mean = st.text_input('smoothness_mean')
    with col3:
        compactness_mean = st.text_input('compactness_mean')
    with col1:    
        concavity_mean = st.text_input('concavity_mean')
    with col2:
        concave_points_mean = st.text_input('concave points_mean')
    with col3:
        symmetry_mean = st.text_input('symmetry_mean')
    with col1:
        fractal_dimension_mean = st.text_input('fractal dimension_mean')
    with col2:
        radius_se = st.text_input('radius_se')
    with col3:
        texture_se = st.text_input('texture_se')
    with col1:
        perimeter_se = st.text_input('perimeter_se')
    with col2:
        area_se = st.text_input('area_se')
    with col3:
        smoothness_se = st.text_input('smoothness_se')
    with col1:
        compactness_se = st.text_input('compactness_se')
    with col2:
        concavity_se = st.text_input('concavity_se')
    with col3:
        concave_points_se = st.text_input('concave points')
    with col1:
        symmetry_se = st.text_input('symmetry_se')
    with col2:
        fractal_dimension_se = st.text_input('fractal dimension_se')
    with col3:
        radius_worst = st.text_input('radius_worst')
    with col1:
        texture_worst = st.text_input('texture_worst')
    with col2:
        perimeter_worst = st.text_input('perimeter_worst')
    with col3:
        area_worst = st.text_input('area_worst')
    with col1:
        smoothness_worst = st.text_input('smoothness_worst')
    with col2:
        compactness_worst = st.text_input('compactness_worst')
    with col3:
        concavity_worst = st.text_input('concavity_worst')
    with col1:
       concave_points_worst = st.text_input('concave points_worst')
    with col2:
        symmetry_worst = st.text_input('Symmetry_w')
    with col3:
        fractal_dimension_worst = st.text_input('fractal dimension_w')



    cancer_diagnosis = ''
    
    #creating a button for prediction 
    if st.button('Cancer Result'):
        cancer_prediction = Breast_cancer.predict([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst,       ]])
        if (cancer_prediction[0] == 1):
          cancer_diagnosis = 'The person is having heart disease'
        else:
          cancer_diagnosis = 'The person does not have any heart disease'
        
    st.success(cancer_diagnosis)
    
    
    
    

    