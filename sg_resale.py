import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report

#download file
df=pd.read_csv(r"C:\Users\kbrad\Downloads\ResaleFlatPricesBasedonApprovalDate19901999.csv")

#SPLIT LOWER AND UPPER BOUND OF STOREY
df[['storey_lower','storey_upper']]=df['storey_range'].str.split('TO',expand=True)

#SPLIT MONTH COLUMN
df[['resale_year','resale_month']]=df['month'].str.split('-',expand=True)

#changing datatype into the correct form
df['resale_year']=pd.to_numeric(df['resale_year'],errors='coerce')
df['resale_month']=pd.to_numeric(df['resale_month'],errors='coerce')
df['storey_lower']=pd.to_numeric(df['storey_lower'],errors='coerce')
df['storey_upper']=pd.to_numeric(df['storey_upper'],errors='coerce')
df['block']=pd.to_numeric(df['block'],errors='coerce')
df['block']=df['block'].fillna(df['block'].mode()[0])

#dropping unnecessary column
df=df.drop('month',axis=1)
df=df.drop('storey_range',axis=1)

#dropping duplicates
df.drop_duplicates(inplace=True)

#category mapping

category_mapping={'1 ROOM':1,'2 ROOM':2,'3 ROOM':3,'4 ROOM':4,'5 ROOM':5,'EXECUTIVE':6,'MULTI GENERATION':7}

#alter the column with mapped numbers
df['flat_type']=df['flat_type'].map(category_mapping)


#Decision tree regressor model

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report


X=df[['town', 'flat_type', 'block', 'street_name', 'floor_area_sqm','flat_model', 'lease_commence_date', 'storey_lower','storey_upper', 'resale_year', 'resale_month']]
y=df['resale_price']

#encoding categorical values
ohe1=OneHotEncoder(handle_unknown='ignore')
ohe1.fit(X[['town']])
X_ohe1=ohe1.fit_transform(X[['town']]).toarray()

ohe2=OneHotEncoder(handle_unknown='ignore')
ohe2.fit(X[['street_name']])
X_ohe2=ohe2.fit_transform(X[['street_name']]).toarray()

ohe3=OneHotEncoder(handle_unknown='ignore')
ohe3.fit(X[['flat_model']])
X_ohe3=ohe3.fit_transform(X[['flat_model']]).toarray()

#independent features after encoding
X=np.concatenate((X[['flat_type', 'block', 'floor_area_sqm', 'lease_commence_date', 'storey_lower','storey_upper', 'resale_year', 'resale_month']].values,X_ohe1,X_ohe2,X_ohe3),axis=1)

scaler=StandardScaler()
X=scaler.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=72)

model1=DecisionTreeRegressor()
model1.fit(X_train,y_train)
train_pred=model1.predict(X_train)
y_pred=model1.predict(X_test)


mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)




st.set_page_config(layout='wide')
st.title(":green[SINGAPORE RESALE FLAT PRICES PREDICTION]")

town_values=['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH','BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI','GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST','KALLANG/WHAMPOA', 'MARINE PARADE', 'QUEENSTOWN', 'SENGKANG','SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN','LIM CHU KANG', 'SEMBAWANG', 'BUKIT PANJANG', 'PASIR RIS']
street_names=['ANG MO KIO AVE 1', 'ANG MO KIO AVE 3', 'ANG MO KIO AVE 4','ANG MO KIO AVE 10', 'ANG MO KIO AVE 5', 'ANG MO KIO AVE 8','ANG MO KIO AVE 6', 'ANG MO KIO AVE 9', 'ANG MO KIO AVE 2','BEDOK RESERVOIR RD', 'BEDOK NTH ST 3', 'BEDOK STH RD',
       'NEW UPP CHANGI RD', 'BEDOK NTH RD', 'BEDOK STH AVE 1','CHAI CHEE RD', 'CHAI CHEE DR', 'BEDOK NTH AVE 4','BEDOK STH AVE 3', 'BEDOK STH AVE 2', 'BEDOK NTH ST 2','BEDOK NTH ST 4', 'BEDOK NTH AVE 2', 'BEDOK NTH AVE 3',
       'BEDOK NTH AVE 1', 'BEDOK NTH ST 1', 'CHAI CHEE ST', 'SIN MING RD','SHUNFU RD', 'BT BATOK ST 11', 'BT BATOK WEST AVE 8','BT BATOK WEST AVE 6', 'BT BATOK ST 21', 'BT BATOK EAST AVE 5',
       'BT BATOK EAST AVE 4', 'HILLVIEW AVE', 'BT BATOK CTRL','BT BATOK ST 31', 'BT BATOK EAST AVE 3', 'TAMAN HO SWEE','TELOK BLANGAH CRES', 'BEO CRES', 'TELOK BLANGAH DR', 'DEPOT RD',
       'TELOK BLANGAH RISE', 'JLN BT MERAH', 'HENDERSON RD', 'INDUS RD','BT MERAH VIEW', 'HENDERSON CRES', 'BT PURMEI RD','TELOK BLANGAH HTS', 'EVERTON PK', 'KG BAHRU HILL', 'REDHILL CL',
       'HOY FATT RD', 'HAVELOCK RD', 'JLN KLINIK', 'JLN RUMAH TINGGI',
       'JLN BT HO SWEE', 'KIM CHENG ST', 'MOH GUAN TER',
       'TELOK BLANGAH WAY', 'KIM TIAN RD', 'KIM TIAN PL', 'EMPRESS RD',
       "QUEEN'S RD", 'FARRER RD', 'JLN KUKOH', 'OUTRAM PK', 'SHORT ST',
       'SELEGIE RD', 'UPP CROSS ST', 'WATERLOO ST', 'QUEEN ST',
       'BUFFALO RD', 'ROWELL RD', 'ROCHOR RD', 'BAIN ST', 'SMITH ST',
       'VEERASAMY RD', 'TECK WHYE AVE', 'TECK WHYE LANE',
       'CLEMENTI AVE 3', 'WEST COAST DR', 'CLEMENTI AVE 2',
       'CLEMENTI AVE 5', 'CLEMENTI AVE 4', 'CLEMENTI AVE 1',
       'WEST COAST RD', 'CLEMENTI WEST ST 1', 'CLEMENTI WEST ST 2',
       'CLEMENTI ST 13', "C'WEALTH AVE WEST", 'CLEMENTI AVE 6',
       'CLEMENTI ST 14', 'CIRCUIT RD', 'MACPHERSON LANE',
       'JLN PASAR BARU', 'GEYLANG SERAI', 'EUNOS CRES', 'SIMS DR',
       'ALJUNIED CRES', 'GEYLANG EAST AVE 1', 'DAKOTA CRES', 'PINE CL',
       'HAIG RD', 'BALAM RD', 'JLN DUA', 'GEYLANG EAST CTRL',
       'EUNOS RD 5', 'HOUGANG AVE 3', 'HOUGANG AVE 5', 'HOUGANG AVE 1',
       'HOUGANG ST 22', 'HOUGANG AVE 10', 'LOR AH SOO', 'HOUGANG ST 11',
       'HOUGANG AVE 7', 'HOUGANG ST 21', 'TEBAN GDNS RD',
       'JURONG EAST AVE 1', 'JURONG EAST ST 32', 'JURONG EAST ST 13',
       'JURONG EAST ST 21', 'JURONG EAST ST 24', 'JURONG EAST ST 31',
       'PANDAN GDNS', 'YUNG KUANG RD', 'HO CHING RD', 'HU CHING RD',
       'BOON LAY DR', 'BOON LAY AVE', 'BOON LAY PL', 'JURONG WEST ST 52',
       'JURONG WEST ST 41', 'JURONG WEST AVE 1', 'JURONG WEST ST 42',
       'JLN BATU', "ST. GEORGE'S RD", 'NTH BRIDGE RD', 'FRENCH RD',
       'BEACH RD', 'WHAMPOA DR', 'UPP BOON KENG RD', 'BENDEMEER RD',
       'WHAMPOA WEST', 'LOR LIMAU', 'KALLANG BAHRU', 'GEYLANG BAHRU',
       'DORSET RD', 'OWEN RD', 'KG ARANG RD', 'JLN BAHAGIA',
       'MOULMEIN RD', 'TOWNER RD', 'JLN RAJAH', 'KENT RD', 'AH HOOD RD',
       "KING GEORGE'S AVE", 'CRAWFORD LANE', 'MARINE CRES', 'MARINE DR',
       'MARINE TER', "C'WEALTH CL", "C'WEALTH DR", 'TANGLIN HALT RD',
       "C'WEALTH CRES", 'DOVER RD', 'MARGARET DR', 'GHIM MOH RD',
       'DOVER CRES', 'STIRLING RD', 'MEI LING ST', 'HOLLAND CL',
       'HOLLAND AVE', 'HOLLAND DR', 'DOVER CL EAST',
       'SELETAR WEST FARMWAY 6', 'LOR LEW LIAN', 'SERANGOON NTH AVE 1',
       'SERANGOON AVE 2', 'SERANGOON AVE 4', 'SERANGOON CTRL',
       'TAMPINES ST 11', 'TAMPINES ST 21', 'TAMPINES ST 91',
       'TAMPINES ST 81', 'TAMPINES AVE 4', 'TAMPINES ST 22',
       'TAMPINES ST 12', 'TAMPINES ST 23', 'TAMPINES ST 24',
       'TAMPINES ST 41', 'TAMPINES ST 82', 'TAMPINES ST 83',
       'TAMPINES AVE 5', 'LOR 2 TOA PAYOH', 'LOR 8 TOA PAYOH',
       'LOR 1 TOA PAYOH', 'LOR 5 TOA PAYOH', 'LOR 3 TOA PAYOH',
       'LOR 7 TOA PAYOH', 'TOA PAYOH EAST', 'LOR 4 TOA PAYOH',
       'TOA PAYOH CTRL', 'TOA PAYOH NTH', 'POTONG PASIR AVE 3',
       'POTONG PASIR AVE 1', 'UPP ALJUNIED LANE', 'JOO SENG RD',
       'MARSILING LANE', 'MARSILING DR', 'MARSILING RISE',
       'MARSILING CRES', 'WOODLANDS CTR RD', 'WOODLANDS ST 13',
       'WOODLANDS ST 11', 'YISHUN RING RD', 'YISHUN AVE 5',
       'YISHUN ST 72', 'YISHUN ST 11', 'YISHUN ST 21', 'YISHUN ST 22',
       'YISHUN AVE 3', 'CHAI CHEE AVE', 'ZION RD', 'LENGKOK BAHRU',
       'SPOTTISWOODE PK RD', 'NEW MKT RD', 'TG PAGAR PLAZA',
       'KELANTAN RD', 'PAYA LEBAR WAY', 'UBI AVE 1', 'SIMS AVE',
       'YUNG PING RD', 'TAO CHING RD', 'GLOUCESTER RD', 'BOON KENG RD',
       'WHAMPOA STH', 'CAMBRIDGE RD', 'TAMPINES ST 42', 'LOR 6 TOA PAYOH',
       'KIM KEAT AVE', 'YISHUN AVE 6', 'YISHUN AVE 9', 'YISHUN ST 71',
       'BT BATOK ST 32', 'SILAT AVE', 'TIONG BAHRU RD', 'SAGO LANE',
       "ST. GEORGE'S LANE", 'LIM CHU KANG RD', "C'WEALTH AVE",
       "QUEEN'S CL", 'SERANGOON AVE 3', 'POTONG PASIR AVE 2',
       'WOODLANDS AVE 1', 'YISHUN AVE 4', 'LOWER DELTA RD', 'NILE RD',
       'JLN MEMBINA BARAT', 'JLN BERSEH', 'CHANDER RD', 'CASSIA CRES',
       'OLD AIRPORT RD', 'ALJUNIED RD', 'BUANGKOK STH FARMWAY 1',
       'BT BATOK ST 33', 'ALEXANDRA RD', 'CHIN SWEE RD', 'SIMS PL',
       'HOUGANG AVE 2', 'HOUGANG AVE 8', 'SEMBAWANG RD', 'SIMEI ST 1',
       'BT BATOK ST 34', 'BT MERAH CTRL', 'LIM LIAK ST', 'JLN TENTERAM',
       'WOODLANDS ST 32', 'SIN MING AVE', 'BT BATOK ST 52', 'DELTA AVE',
       'PIPIT RD', 'HOUGANG AVE 4', 'QUEENSWAY', 'YISHUN ST 61',
       'BISHAN ST 12', "JLN MA'MOR", 'TAMPINES ST 44', 'TAMPINES ST 43',
       'BISHAN ST 13', 'JLN DUSUN', 'YISHUN AVE 2', 'JOO CHIAT RD',
       'EAST COAST RD', 'REDHILL RD', 'KIM PONG RD', 'RACE COURSE RD',
       'KRETA AYER RD', 'HOUGANG ST 61', 'TESSENSOHN RD', 'MARSILING RD',
       'YISHUN ST 81', 'BT BATOK ST 51', 'BT BATOK WEST AVE 4',
       'BT BATOK WEST AVE 2', 'JURONG WEST ST 91', 'JURONG WEST ST 81',
       'GANGSA RD', 'MCNAIR RD', 'SIMEI ST 4', 'YISHUN AVE 7',
       'SERANGOON NTH AVE 2', 'YISHUN AVE 11', 'BANGKIT RD',
       'JURONG WEST ST 73', 'OUTRAM HILL', 'HOUGANG AVE 6',
       'PASIR RIS ST 12', 'PENDING RD', 'PETIR RD', 'LOR 3 GEYLANG',
       'BISHAN ST 11', 'PASIR RIS DR 6', 'BISHAN ST 23',
       'JURONG WEST ST 92', 'PASIR RIS ST 11', 'YISHUN CTRL',
       'BISHAN ST 22', 'SIMEI RD', 'TAMPINES ST 84', 'BT PANJANG RING RD',
       'JURONG WEST ST 93', 'FAJAR RD', 'WOODLANDS ST 81',
       'CHOA CHU KANG CTRL', 'PASIR RIS ST 51', 'HOUGANG ST 52',
       'CASHEW RD', 'TOH YI DR', 'HOUGANG CTRL', 'KG KAYU RD',
       'TAMPINES AVE 8', 'TAMPINES ST 45', 'SIMEI ST 2',
       'WOODLANDS AVE 3', 'LENGKONG TIGA', 'WOODLANDS ST 82',
       'SERANGOON NTH AVE 4', 'SERANGOON CTRL DR', 'BRIGHT HILL DR',
       'SAUJANA RD', 'CHOA CHU KANG AVE 3', 'TAMPINES AVE 9',
       'JURONG WEST ST 51', 'YUNG HO RD', 'SERANGOON AVE 1',
       'PASIR RIS ST 41', 'GEYLANG EAST AVE 2', 'CHOA CHU KANG AVE 2',
       'KIM KEAT LINK', 'PASIR RIS DR 4', 'PASIR RIS ST 21',
       'SENG POH RD', 'HOUGANG ST 51', 'JURONG WEST ST 72',
       'JURONG WEST ST 71', 'PASIR RIS ST 52', 'TAMPINES ST 32',
       'CHOA CHU KANG AVE 4', 'CHOA CHU KANG LOOP', 'JLN TENAGA',
       'TAMPINES CTRL 1', 'TAMPINES ST 33', 'BT BATOK WEST AVE 7',
       'JURONG WEST AVE 5', 'TAMPINES AVE 7', 'WOODLANDS ST 83',
       'CHOA CHU KANG ST 51', 'PASIR RIS DR 3', 'YISHUN CTRL 1',
       'CHOA CHU KANG AVE 1', 'WOODLANDS ST 31', 'BT MERAH LANE 1',
       'PASIR RIS ST 13', 'ELIAS RD', 'BISHAN ST 24', 'WHAMPOA RD',
       'WOODLANDS ST 41', 'PASIR RIS ST 71', 'JURONG WEST ST 74',
       'PASIR RIS DR 1', 'PASIR RIS ST 72', 'PASIR RIS DR 10',
       'CHOA CHU KANG ST 52', 'CLARENCE LANE', 'CHOA CHU KANG NTH 6',
       'PASIR RIS ST 53', 'CHOA CHU KANG NTH 5', 'ANG MO KIO ST 21',
       'JLN DAMAI', 'CHOA CHU KANG ST 62', 'WOODLANDS AVE 5',
       'WOODLANDS DR 50', 'CHOA CHU KANG ST 53', 'TAMPINES ST 72',
       'UPP SERANGOON RD', 'JURONG WEST ST 75', 'STRATHMORE AVE',
       'ANG MO KIO ST 31', 'TAMPINES ST 34', 'YUNG AN RD',
       'WOODLANDS AVE 4', 'CHOA CHU KANG NTH 7', 'ANG MO KIO ST 11']
flat_model_values=['IMPROVED', 'NEW GENERATION', 'MODEL A', 'STANDARD', 'SIMPLIFIED','MODEL A-MAISONETTE', 'APARTMENT', 'MAISONETTE', 'TERRACE','2-ROOM', 'IMPROVED-MAISONETTE', 'MULTI GENERATION','PREMIUM APARTMENT']
lease_commence_date_values=[1977, 1976, 1978, 1979, 1984, 1980, 1985, 1981, 1982, 1986, 1972,1983, 1973, 1969, 1975, 1971, 1974, 1967, 1970, 1968, 1988, 1987,1989, 1990, 1992, 1993, 1994, 1991, 1995, 1996, 1997]
storey_lower_values=[10,  4,  7,  1, 13, 19, 16, 25, 22]
storey_upper_values=[12,  6,  9,  3, 15, 21, 18, 27, 24]
with st.form("Data"):
        col1,col2=st.columns(2)
        with col1:
            st.write(' ')
            town=st.selectbox("Town:",town_values,key=1)
            street_name=st.selectbox("Street_name",street_names,key=2)
            flat_model=st.selectbox("Flat_model",flat_model_values,key=3)
            lease_commence_date=st.selectbox("Lease year",lease_commence_date_values,key=4)
            storey_lower=st.selectbox("Storey lower",storey_lower_values,key=5)
            storey_upper=st.selectbox("Storey upper",storey_upper_values,key=6)
        with col2:
            flat_type= st.text_input("Enter flat_type(Min:1,Max:7)")
            block= st.text_input("Enter block(Min:1.0,Max:980.0)")
            floor_area_sqm=st.text_input("Enter floor area(Min:28.0,Max:180.5)")
            resale_year=st.text_input("Enter resale year(Min:1990,Max:1999)")
            resale_month=st.text_input("Enter resale month(Min:1,Max:12)")
            submit_button=st.form_submit_button(label="PREDICT RESALE PRICE")
    
            if submit_button:
                 
                import pickle
                with open(r"C:\Users\kbrad\OneDrive\Documents\New folder\sg_model.pkl",'rb') as file:
                    loaded_model=pickle.load(file)

                with open(r"C:\Users\kbrad\OneDrive\Documents\New folder\sg_scaler.pkl",'rb') as file1:
                    loaded_scalar=pickle.load(file1)

                with open(r"C:\Users\kbrad\OneDrive\Documents\New folder\sg_town.pkl",'rb') as file2:
                    loaded_town=pickle.load(file2)
                with open(r"C:\Users\kbrad\OneDrive\Documents\New folder\sg_street.pkl",'rb') as file3:
                    loaded_street=pickle.load(file3)
                with open(r"C:\Users\kbrad\OneDrive\Documents\New folder\sg_flat.pkl",'rb') as file4:
                    loaded_flat=pickle.load(file4)

                user_input=np.array([[town,flat_type,block,street_name, floor_area_sqm,flat_model,lease_commence_date,storey_lower,storey_upper, resale_year, resale_month]])
                user_input_ohe1=loaded_town.transform(user_input[:,[0]]).toarray()
                user_input_ohe2=loaded_street.transform(user_input[:,[3]]).toarray()
                user_input_ohe3=loaded_flat.transform(user_input[:,[5]]).toarray()

                user_input=np.concatenate((user_input[:,[1,2,4,6,7,8,9,10]],user_input_ohe1,user_input_ohe2,user_input_ohe3),axis=1)
                user_input1=loaded_scalar.transform(user_input)
                user_prediction=loaded_model.predict(user_input1)
                st.write(":blue[Predicted resale price:]",user_prediction)

with st.sidebar:
    st.title(":blue[SINGAPORE RESALE FLAT PRICE PREDICTION]")
    st.header(":red[STEPS FOLLOWED]")
    st.caption("Downloaded the data set and detected outliers in the dataset")
    st.caption("Transformed the data into a suitable format and perform any necessary cleaning and pre-processing steps")
    st.caption("Created ML Regression model which predicts continuous variable ‘Resale flat_Price’")
    
    st.caption("Created a streamlit page where you can insert each column value and you will get the Resale_Price predicted value ")
   
    st.header(":red[TECHNOLOGIES USED]")
    st.caption("Python scripting,Pandas,Numpy,Seaborn,Matplotlib,Data Preprocessing,EDA, Streamlit")






