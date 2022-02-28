
from pywebio.input import input, FLOAT, NUMBER, TEXT
from pywebio.output import put_text, put_markdown, put_loading, put_collapse, put_table, toast
import pandas as pd
import pywebio

x_symptoms = []

def symp_check(x):
    if x not in x_symptoms:
        return 'Try again !!'

def check_age(age):
    if age > 100:
        return 'Too old'
    elif age < 8:
        return 'Too young'

def count_check(c):
    if c > 17:
        return "Too many symptoms"

def modelling():
    
    df = pd.read_csv("dataset.csv")
    inputs = df.drop('Disease',axis='columns')
    target = df['Disease']
    
    col_names = []
    for col in inputs.columns:
        col_names.append(col)
        
    symptoms = []
    for col in col_names:
        new = df.drop_duplicates(subset = [col])
        for ind in new.index:
            symptom = new[col][ind]
            if symptom not in symptoms and str(symptom) != "nan": #
                symptoms.append(symptom)
    
    print("Please wait !!")
    print("Loading.....")
    symptoms.append("Disease")

    new_df = pd.DataFrame(columns = symptoms)
    
    for ind in df.index:
        smp = [0]*(len(symptoms)-1)
        for symp in col_names:
            val = df[symp][ind]
            if str(val) == "nan": #
                continue
            idx = symptoms.index(val)
            smp[idx] = 1
            
        smp.append(df["Disease"][ind])
        #print(len(smp),smp)
        new_df.loc[ind] = smp
    print("Dataset loaded...")
    
    #inputs2 = new_df.drop('Disease',axis='columns')
    #target2["Disease"] = new_df['Disease']
    
    from sklearn.preprocessing import LabelEncoder
    le_disease = LabelEncoder()
    
    new_df["Diseases"] = le_disease.fit_transform(new_df['Disease'])
    
    inputs2 = new_df.drop('Disease',axis='columns')
    
    target2 = new_df['Diseases']
    
    inputs2 = inputs2.drop('Diseases',axis=1)
    
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(inputs2,target2,test_size=0.3)
    
    print("Test and Train data created")
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(X_train,y_train)
    model.score(X_test,y_test)
    print("Model fitted ")
    print("Acurracy is ",model.score(X_test,y_test))
    desc = pd.read_csv("symptom_description.csv")
    desc.head()
    
    sev = pd.read_csv("symptom_severity.csv")
    sev.head()
    
    prec = pd.read_csv( "symptom_precaution.csv")
    prec.head()
    
    for ind,smp in enumerate(symptoms):
        try:
            symptoms[ind] = smp.strip(" ")
        except:
            pass
    return symptoms, model, desc, sev, prec, col_names, le_disease

def predict(symptoms, model, desc, sev, prec, col_names, le_disease):
    print(len(symptoms))
    prd = [0]*(len(symptoms) - 1)
    
    name = input("Enter your name ",type = TEXT,required =True)
    age = input("enter your age", type = NUMBER, validate = check_age,required =True)

    no_smp = input("How many symptoms do you have ?",type = NUMBER,validate = count_check,required =True)
    #pred = pd.DataFrame(prd).T
    #smps = ["headache","joint pain","dehydration","itching"]
    smps = []
    global x_symptoms
    x_symptoms = symptoms.copy()
    x_symptoms.remove('Disease')
    x_symptoms.append('None')
    print(symptoms)
    for j in range(no_smp):
        #ss = input("Symptom "+str(j+1), type = TEXT)
        ss = input(label = "Symptom "+str(j+1), datalist = x_symptoms, validate = symp_check, required  = True)
        #put_text(ss)
        smps.append(ss)

    for smp in smps:
        #smp = smp.replace(" ","_")
        if smp in symptoms:
            ind = symptoms.index(smp)
            prd[ind] = 1
    if 'None' in smps:
        smps.remove('None')
    #print(prd)
    result = model.predict([prd])
    ds = le_disease.inverse_transform([result])[0]
    
    x = desc.loc[desc['Disease'] == ds]
    idx = x.index[0]
    put_markdown(r""" # Results
    """)
    name = "Name : " + name
    age = "Age : " + str(age)

    put_text(name)
    put_text(age)
    #put_text("Desciptions :")
    #put_text(ds)
    #put_text(desc["Description"][idx])

    put_collapse('Disease : ' + ds, desc["Description"][idx], open = True)

    sm = 0
    for i in smps:
        #i = i.replace(" ","_")
        print(i in symptoms)
        x = sev.loc[sev['Symptom'] == i]
        idx = x.index[0]
        #print(idx)
        sm += sev["weight"][idx]
    wg = sm/len(smps)

    if wg <=2 :
        msg = "Dont panic it's just a normal symptoms and can be cured easily"
        toast( msg, position='right', color='#28ff21', duration=0)
    elif wg <=4 and wg > 2:
        msg = "The symptoms are not normal, visit doctor as soon as possible"
        toast( msg, position='right', color='#f8ff21', duration=0)
    else:
        msg = "You are at a high risk !!"
        toast( msg, position='right', color='#ff2121', duration=0)

    col_names = []
    for col in prec.columns:
        col_names.append(col)
    precautions = []
    for indx,i in enumerate(col_names):
        x = prec.loc[prec['Disease'] == ds]
        idx = x.index[0]
        #print(idx)
        sm = prec[i][idx]
        #put_text(sm)
        precautions.append([indx + 1, sm])
    put_collapse('Precautions ', [ put_table([ ['S.No', 'Precaution'], precautions[0],precautions[1],precautions[2],precautions[3], ])], open=True)

if __name__ == '__main__':
    symptoms, model, desc, sev, prec, col_names, le_disease = modelling()
    flag = True
    pos = ['Yes','No']
    while flag:
        result = predict(symptoms, model, desc, sev, prec, col_names, le_disease)
        inp = input(label = "Want to check your disease  again ? ", datalist = pos, required  = True)
        inp = inp.lower()
        if inp == 'yes':
            print(111111)
            continue
        elif  inp == 'no':
            break
        else:
            put_text('Invalid input !!')
        put_text("-"*50)


