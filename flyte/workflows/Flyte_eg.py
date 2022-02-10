from flytekit import workflow,task
import pandas as pd
from . import utility 
import warnings
import matplotlib.pyplot as plt
@task
def data_read() -> pd.DataFrame:
    '''
    This function reads clean data from git repo
    '''
    print("Reading clean data\n")
    df = pd.read_csv("https://raw.githubusercontent.com/bharatbajoria/Projects/main/Data_csv.csv",index_col=0 )
    return df
@task
def stat_test(df: pd.DataFrame):
    import utility
    print("Running statistical tests\n")
    utility.kpss_test(df['Category A'], cat = 'Category A')
    print("\n")
    utility.kpss_test(df['Category B'], cat = 'Category B')
    print("\n")
    utility.kpss_test(df['Category C'], cat = 'Category C')
@task
def model_a(df: pd.DataFrame):
    print("\nTraining Model A")
    df_A = utility.series_to_supervised(list(df['Category A'].copy()), n_in=2, n_out=1, dropnan=True)
    df_A['Offer'] = df.loc[list(df_A.index)]['Offer']
    df_A['Festival'] = df.loc[list(df_A.index)]['FESTIVAL'].map(lambda x: 0 if x =='No-Festival' else 1)
    df_A = df_A[['x-1(t-2)', 'x-1(t-1)' , 'Offer', 'Festival','Y1(t)']]
    print(df_A.head())
    
    train_X, train_y, test_X, test_y = utility.train_test(df_A,n_train=330)
    history,model = utility.model_Category_A(train_X, train_y, test_X, test_y)
    
    yhat = model.predict(test_X)
    yhat_train = model.predict(train_X)
    warnings.filterwarnings('ignore')
    y = [i[0] for i in yhat]
    y_train = [i[0] for i in yhat_train]
    print(utility.MAPE(list(test_y),y),utility.MAPE(list(train_y),y_train))

    plt.plot(y_train[:] + y[:], label = 'Predicted')
    plt.plot(df['Category A'], label = 'True')
    plt.legend()
    
@task
def model_b(df: pd.DataFrame):
    print("\nTraining Model B")
    df_B = utility.series_to_supervised(list(df['Category B'].copy()), n_in=2, n_out=1, dropnan=True)
    df_B['Offer'] = df.loc[list(df_B.index)]['Offer']
    df_B['Festival'] = df.loc[list(df_B.index)]['FESTIVAL'].map(lambda x: 0 if x =='No-Festival' else 1)
    df_B = df_B[['x-1(t-2)', 'x-1(t-1)' , 'Offer', 'Festival','Y1(t)']]
    df_B.head()
    
    train_X, train_y, test_X, test_y = utility.train_test(df_B,n_train=330)
    history,model = utility.model_Category_B(train_X, train_y, test_X, test_y)
    
    model_b = model
    
    yhat = model_b.predict(test_X)
    yhat_train = model_b.predict(train_X)
    warnings.filterwarnings('ignore')
    y = [i[0] for i in yhat]
    y_train = [i[0] for i in yhat_train]
    print(utility.MAPE(list(test_y),y),utility.MAPE(list(train_y),y_train))

    plt.plot(y_train[:] + y[:], label = 'Predicted')
    plt.plot(df['Category B'], label = 'True')
    plt.legend()
    
@task
def model_c(df: pd.DataFrame):
    print("\nTraining Model C")
    df_C = utility.series_to_supervised(list(df['Category C'].copy()), n_in=2, n_out=1, dropnan=True)
    df_C['Offer'] = df.loc[list(df_C.index)]['Offer']
    df_C['Festival'] = df.loc[list(df_C.index)]['FESTIVAL'].map(lambda x: 0 if x =='No-Festival' else 1)
    df_C = df_C[['x-1(t-2)', 'x-1(t-1)' , 'Offer', 'Festival','Y1(t)']]
    df_C.head()
    
    train_X, train_y, test_X, test_y = utility.train_test(df_C,n_train=330)
    
    history, model = utility.model_Category_C(train_X, train_y, test_X, test_y)
    
    model_c = model
    
    yhat = model_c.predict(test_X)
    yhat_train = model_c.predict(train_X)
    warnings.filterwarnings('ignore')
    y = [i[0] for i in yhat]
    y_train = [i[0] for i in yhat_train]
    print(utility.MAPE(list(test_y),y),utility.MAPE(list(train_y),y_train))

    plt.plot(y_train[:] + y[:], label = 'Predicted')
    plt.plot(df['Category C'], label = 'True')
    plt.title("Category C : Predicted Vs True")
    plt.legend()
    plt.show()
@workflow
def pipeline():
    df = data_read()
    stat_test(df=df)
    model_a(df=df)
    model_b(df=df)
    model_c(df=df)
    
if __name__=="__main__":
    print(f"Run",pipeline())
    