import streamlit as st
import pickle
import pandas as pd

filename = 'model.pkl'
filename2 = 'label_encoder.pkl'
loaded_model = pickle.load(open(filename, 'rb'))
enc = pickle.load(open(filename2, 'rb'))

file = open('Datasets/symptoms.txt', 'r')
symptoms = [x.strip() for x in file.readline()[1:-1].replace("'", "").split(",")]
description_df = pd.read_csv("Datasets/symptom_Description.csv")
severity_df = pd.read_csv("Datasets/symptom_severity.csv")
precaution_df = pd.read_csv( "Datasets/symptom_precaution.csv")

import pandas as pd

def predict(inp):
    inps = [0] * len(symptoms)
    for symp in inp:
        inps[symptoms.index(symp)] = 1

    predicted_disease = enc.inverse_transform(loaded_model.predict([inps]))[0]

    description = description_df.loc[description_df['Disease'] == predicted_disease, 'Description'].values[0]
    precaution = precaution_df.loc[precaution_df['Disease'] == predicted_disease, ['Precaution_1', 'Precaution_2', 'Precaution_3']].values[0]

    return predicted_disease, description, precaution


def main():
    st.image('Datasets/image.jpg', width=700)

    st.title('Symptom Prediction')
    total_symptoms = st.number_input('How many symptoms do you have?', min_value=1, max_value=len(symptoms), step=1, value=1)
    if total_symptoms > 10:
        st.warning("Number of Symptoms are too high")
    selected_symptoms = []
    if total_symptoms <= 10 :
        for i in range(total_symptoms):
            symptom_input = st.selectbox(f'Symptom {i+1}', symptoms)
            selected_symptoms.append(symptom_input)

        if st.button('Predict'):
            predicted_disease, description, precaution = predict(selected_symptoms)
            st.write('Predicted Disease:', predicted_disease)
            st.write('Description:', description)
            
            prec = pd.DataFrame(precaution, columns=['Precautions'])
            st.write(prec)

            sev = 0
            for symp in selected_symptoms:
                sev += severity_df.loc[severity_df['Symptom'] == symp, 'weight'].values
            
            sev = sev/len(selected_symptoms)
            print(sev)
            if sev <= 2.5:
                msg = "Dont panic it's just a normal symptoms and can be cured easily."
            elif sev <= 4 :
                msg = "The symptoms are not normal, visit doctor whenever possible !!"
            else:
                msg = "You are at a high risk, visit doctor as soon as possible !!"
            st.warning(msg)

if __name__ == '__main__':
    main()
