
from pywebio.input import input, FLOAT, NUMBER, TEXT
from pywebio.output import put_text, put_markdown, put_loading, put_collapse, put_table, toast
import pandas as pd
import pywebio
import math 
import numpy as np

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

#def tree(x_train,x_test):

def infogain(right,left, diseases, index, Entropy,c):
        rc = len(right)
        lc = len(left)
        r_sm = 0
        l_sm = 0
        for i in diseases:
            #print(i, end = " ")
            new = left["prognosis"] == i
            new = list(new)
            cr = new.count(True)
            new = right["prognosis"] == i
            new = list(new)
            cl = new.count(True)
            #print() 
            
            if rc != 0 and lc != 0: # -------------------------------------- doubt -------------------------
                if cl == 0:
                    l_sm = 0
                else:
                    l_sm = cl/lc * math.log(cl/lc)
                if cr == 0:
                    r_sm = 0
                else:
                    r_sm = cr/rc * math.log(cr/rc) 
            
            elif rc == 0:
                r_sm = 0 
                if cl == 0:
                    l_sm = 0
                else:
                    l_sm = cl/lc * math.log(cl/lc)
                
            elif lc == 0:
                l_sm = 0
                if cr == 0:
                    r_sm = 0
                else:
                    r_sm = cr/rc * math.log(cr/rc)
            
        r_sm = (rc / index) * r_sm
        l_sm = (lc / index) * l_sm
        ig = Entropy + (r_sm + l_sm)
        return ig

def predict2(arr,train,col_disease,test,flag,zsmp):
    #print(arr , end =" " )
    if list(arr) == [] or not flag:
        #return train
        sm = []
        mx = 0
        for i in col_disease:
            #print(i, end = " ")
            new = train["prognosis"] == i
            new = list(new)
            c = new.count(True)
            if mx < c:
                label = i
                mx = c
            #print(c)
            #sm = sm + c/index * math.log(c/index) 
            #sm.append(c)
        return label
    elif flag:
        split = arr[len(arr)-1]
        dsx = zsmp[split]
        #print(split,dsx)
        #print(len(train),dsx)
        z_train = train
        if test[split] > 0.5:
            train = train[train[dsx] > 0.5]
        elif test[split] < 0.5:
            train = train[train[dsx] < 0.5]
        arr = np.delete(arr, len(arr)-1)
        
        if len(train) == 0:
            train = z_train
            flag = False
        
        return predict2(arr,train,col_disease,test,flag,zsmp)

def accuracy(x_test,y_test,prdt,train_df,diseases,zsmp,le_disease):  
    count = 0
    t_count = 0
    for row in x_test.index:
        prd = np.array(x_test.iloc[row])
        #print(row)
        prd_lb = predict2(prdt,train_df,diseases,prd, True,zsmp)
        lb = y_test.iloc[row]
        if (le_disease.inverse_transform([lb])[0] == prd_lb):
            count += 1 
        t_count += 1
        print("Accuracy is ",count / t_count)


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
    new_df  = new_df.rename(columns={"Disease":"prognosis"})
    print(new_df.columns)
    perc = len(new_df) * 0.3
    train_df = new_df.iloc[ : int(perc)]
    test_df = new_df.iloc[int(perc) : ]
    
    from sklearn.preprocessing import LabelEncoder
    le_disease = LabelEncoder()
    
    train_df["Diseases"] = le_disease.fit_transform(train_df['prognosis'])
    test_df["Diseases"] = le_disease.fit_transform(test_df['prognosis'])
    disease = train_df.drop_duplicates(subset = ["prognosis"])

    dis = []
    for ind in disease.index:
        di = disease["prognosis"][ind]
        dis.append(di)

    diseases = dis

    index = len(train_df.index)
    x = list(train_df.columns)
    
    sm = 0
    for i in diseases:
        #print(i, end = " ")
        new = train_df["prognosis"] == i
        new = list(new)
        c = new.count(True)
        sm = sm + c/index * math.log(c/index) 

    Entropy = -sm
    zsmp = list(train_df.columns)

    cj = zsmp.copy()
    cj.remove("prognosis")
    cj.remove("Diseases")
    igs = []
    count = 0
    for j in cj :
        right = train_df[train_df[j] > 0.5]
        left = train_df[train_df[j] < 0.5]
        print(len(right),len(left) ,end = " ")
        ig = infogain(right,left,diseases, index, Entropy,c)
        igs.append(ig)
        print(count,ig)
        count += 1

    prdt = np.argsort(igs)

    x_train = train_df.drop('Diseases',axis=1)
    x_train = x_train.drop('prognosis',axis=1)
    y_train = train_df["Diseases"]

    x_test = test_df.drop('Diseases',axis=1)
    x_test = x_test.drop('prognosis',axis=1)
    y_test = test_df["Diseases"]
    print("Test and Train data created")
    #x_test,y_test,prdt,train_df,diseases,zsmp,le_disease
    #print("Acurracy is ",accuracy(x_test,y_test,prdt,train_df,diseases,zsmp,le_disease))
    print("Model fitted ")

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
    model = 0
    return symptoms, model, desc, sev, prec, col_names, le_disease, train_df, diseases, prdt, zsmp



def predict(symptoms, model, desc, sev, prec, col_names, le_disease, prdt, train_df, diseases, zsmp ):
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
    #result = model.predict([prd])
    result = predict2(prdt, train_df, diseases, prd, True, zsmp)
    #ds = le_disease.inverse_transform([result])[0]
    ds = result
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
    symptoms, model, desc, sev, prec, col_names, le_disease, train_df, diseases, prdt, zsmp = modelling()
    flag = True
    pos = ['Yes','No']
    while flag:
        result = predict(symptoms, model, desc, sev, prec, col_names, le_disease, prdt, train_df, diseases, zsmp)
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
