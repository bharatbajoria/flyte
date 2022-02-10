from datetime import datetime,date,timedelta
from statsmodels.tsa.stattools import kpss
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import numpy as np

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow



warnings.filterwarnings('ignore')

def update_Medium_Promotions_Offerdays(promotions,df):
    promo_dict = dict(zip(promotions['Promotions'], promotions['Offer days']))
    promo_dict['NA'] = 0
    df['Offer days'] = df['Promo'].map(promo_dict)

    medium_dict = dict(zip(promotions['Promotions'], promotions['Medium']))
    promo_dict['NA'] = 0
    df['Medium'] = df['Promo'].map(medium_dict)

    df['Promotions'] =df['Promo']

    df.drop(['Promo'], inplace = True, axis = 1)
    
    return df

    

def get_fillna(df):
    '''
    Function to fill null values with appropriate values
    '''
    
    df['FESTIVAL'].fillna('No-Festival', inplace = True)
    df['RELIGION'].fillna('NA', inplace = True)
    df['FESTIVAL'].fillna('NA', inplace = True)
    df['COUNTRY'].fillna('NA', inplace = True)
    df['Medium'].fillna('NA', inplace = True)
    df['Promo'].fillna('NA', inplace = True)
    df['End Date'].fillna('NA', inplace = True)
    
    return df


def get_offer_dates(promotions):
    '''
    Function to get dates between end date and start date on which offer is valid
    '''
    
    sdate = promotions['Date'].iloc[0]
    edate = promotions['End Date'].iloc[0]
    offer = promotions['Promotions'].iloc[0]


    all_offer_dicts = offer_date_map(dates_bw_2dates(sdate,edate),offer)

    for i in range(1,len(promotions)):

        sdate = promotions['Date'].iloc[i]
        edate = promotions['End Date'].iloc[i]
        offer = promotions['Promotions'].iloc[i]

        current_offer = offer_date_map(dates_bw_2dates(sdate,edate),offer)

        all_offer_dicts = {**all_offer_dicts,**current_offer}


    df_offer_dates = pd.DataFrame.from_dict([list(all_offer_dicts.keys()),list(all_offer_dicts.values())]).T
    df_offer_dates.columns = ['Date','Promo']
    return df_offer_dates

# month_dict is a doctionary mapping quarters awith months 
month_dict = {}
for i in range(1,13):
    if i<4:
        month_dict[i] = 4
    elif i<7:
        month_dict[i] = 1
    elif i<10:
        month_dict[i] = 2
    else:
        month_dict[i] = 3


def date_cleaning(festival,sales,promotions):
    '''
    Function to clean dates and create a common format across all dataframes
    '''
    
    festival['DATE'] = festival['DATE'].map(lambda x: x.replace(".","/")) 
    sales['Date'] = sales['Date'].map(lambda x: x.replace("/20","/2020"))
    promotions['End Date'] = promotions['End Date'].map(lambda x: x.replace("/20","/2020"))
    promotions['Start Date'] = promotions['Start Date'].map(lambda x: x.replace("/20","/2020"))
    
    return festival,sales,promotions


def day_format(a):
    '''
    Function to have dates in a format appropriate to input to datetime library
    '''
    
    a = a.split("/")    
    return date(int(a[2]) ,int(a[1]) ,int(a[0]))    
    

def days(a,b):
    '''
    Function to find no. of days between 2 dates
    '''
    
    a = day_format(a)
    b = day_format(b)
    #date_format = "%d/%m/%Y"
    #a = datetime.strptime(a, date_format)
    #b = datetime.strptime(b, date_format)
    delta = b - a
    return delta.days

def date_modification(date_modified):
    '''
    Function to modify dates and have them in a common format
    '''
    
    if date_modified.day<10:
        date = '0' +str(date_modified.day) +"/"
    else:
        date = str(date_modified.day) +"/"
    if date_modified.month<10:
        date+= '0' +str(date_modified.month) +"/"
    else:
        date+= str(date_modified.month) +"/"
    date+=str(date_modified.year)
    
    return date

    
#define KPSS
# Null : Stationary
def kpss_test(timeseries, cat = 'Category A'):
    '''
    Function to do KPSS test for stationarity
    '''
    
    print ('Results of KPSS Test:',cat)
    kpsstest = kpss(timeseries, regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    print(kpss_output)
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
def plot_bar(x,y,title):
    '''
    Function to plot barchart
    '''
    
    plt.bar(x,y, width=0.5)
    plt.xticks(rotation = 90)
    plt.title(title)
    plt.show()

    
def dates_bw_2dates(sdate,edate):
    '''
    Function to find dates between 2 dates
    '''
    
    sdate = day_format(sdate)   # start date
    edate = day_format(edate)   # end date
    date_modified=sdate
    list1=[date_modification(sdate)] 


    while date_modified<edate:
        date_modified+=timedelta(days=1) 
        #print(date_modified.date())
        list1.append(date_modification(date_modified))

    return list1 
def offer_date_map(date_list,offer):
    '''
    Function to map dates with offer on which they are valid
    '''
    
   # print(date_list)
    offer_map = {}
    for i in date_list:
        offer_map[i] = offer
    return offer_map


def train_test(data,n_train=330):
    '''
    Function to create train-test split for time series data
    '''
    
    df_A = data.copy()
    values = df_A.values
    n_train = 330
    train = values[:n_train, :]
    test = values[n_train:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
   # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    return train_X, train_y, test_X, test_y

from pandas import concat


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    '''
    Function to create timeseries features by shifting them 
    '''
    
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('x-%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('Y%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('Y%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
# drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def get_y(model,test_X,train_X):
    '''
    Function to ease redundancy of same code for getting Y values of train and test for different models
    '''
    
    yhat = model.predict(test_X)
    yhat_train = model.predict(train_X)
    y = [i[0] for i in yhat]
    y_train = [i[0] for i in yhat_train]
    return y,y_train    

def MAPE(y_true, y_pred): 
    '''
    Function to find Mean Absolute Percentage Error
    '''
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def model_Category_A(train_X, train_y, test_X, test_y):
    '''
    Function to train Model on Category A
    '''
    
    model =  tensorflow.keras.Sequential()
    model.add(LSTM(120, return_sequences = True,input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(LSTM(120,return_sequences=True))
    model.add(LSTM(90,return_sequences=False))
    model.add(Dense(1))
    model.compile(loss=tensorflow.keras.losses.MeanAbsolutePercentageError(),
                  optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=450, batch_size=75, validation_data=(test_X, test_y), verbose=0, shuffle=False)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    
    return history,model

def model_Category_B(train_X, train_y, test_X, test_y):
    '''
    Function to train Model on Category B
    '''
    
    model =  tensorflow.keras.Sequential()
    model.add(LSTM(180, return_sequences = True,input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(LSTM(180,return_sequences=True))
    model.add(LSTM(120,return_sequences=False))
    model.add(Dense(1))
    model.compile(loss=tensorflow.keras.losses.MeanAbsolutePercentageError(),
                  optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=1050, batch_size=60, validation_data=(test_X, test_y), verbose=0, shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    
    return history, model

def model_Category_C(train_X, train_y, test_X, test_y):
    '''
    Function to train Model on Category C
    '''
    
    model =  tensorflow.keras.Sequential()
    model.add(LSTM(180, return_sequences = True,input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(LSTM(180,return_sequences=True))
    model.add(LSTM(120,return_sequences=False))
    model.add(Dense(1))
    model.compile(loss=tensorflow.keras.losses.MeanAbsolutePercentageError(),
                  optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=2000, batch_size=60, validation_data=(test_X, test_y), verbose=0, shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    return history, model

def prediction_future(model,df,var = 'Category A', days = 7 ):
    future_x = list(df.iloc[365:][var])
    future_fest = [1,0,0,0,0,0,0]
    future_offer = [0,0,0,0,0,0,0]

    for i in range(days):
       # print(i)
        x1 = np.array([[future_x[-2:] + [future_offer[i] , future_fest[i]]]])  
        
        future_x.append(model.predict(x1)[0][0])
        #print(future_x)
    return future_x[2:]
