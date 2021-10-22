import datetime as dt
import numpy as np
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
from matplotlib import style
import os
import bs4 as bs
import pickle
import requests

# ================================= functions ================================

def save_sp500_tickers():
    resp=requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup=bs.BeautifulSoup(resp.text,'lxml')
    table=soup.find('table',{'class':'wikitable sortable'})
    tickers=[]
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.replace('\n','')
        if "." in ticker:
            ticker = ticker.replace('.','-')
        tickers.append(ticker)
    
    file=open('sp500tickers','wb')
    pickle.dump(tickers,file)
    file.close()
    return(tickers)

def get_data_from_yahoo(start,end):
 
    file=open('sp500tickers','rb')
    tickers=pickle.load(file)
    file.close()
    for ticker in tickers:
        print(ticker)
        df=web.DataReader(ticker,'yahoo',start,end)
        df.to_csv('yahoo_data/{}.csv'.format(ticker))
        
def compile_data():
    file=open('sp500tickers','rb')
    tickers=pickle.load(file)
    file.close()
    main_df=pd.DataFrame()
    for count,ticker in enumerate(tickers):
        # print(ticker)
        df=pd.read_csv('yahoo_data/{}.csv'.format(ticker))
        df.set_index('Date',inplace=True)
        df.rename(columns={'Adj Close' : ticker}, inplace=True)
        df.drop(['Open','High','Low','Close','Volume'],1,inplace=True)
        
        if main_df.empty:
            main_df=df
        else:
            main_df=main_df.join(df,how='outer')
        if count%10==0:
            print(count)
            
    main_df.to_csv('sp500.csv')
    



# ============================================================================
  
# file=open('sp500tickers','rb')
# tickers=pickle.load(file)
# file.close()

# Get data

# start=dt.datetime(2017,1,1)
# end=dt.datetime(2021,1,29)
# get_data_from_yahoo(start, end)

# compile_data()
    
sp500=pd.read_csv('yahoo_data/sp500.csv', parse_dates=True,index_col=0)

tickers=sp500.columns.values.tolist()

df_corr=sp500.corr()

df_corr_num=df_corr.set_index(pd.Index(np.arange(0,df_corr.shape[0]).tolist()))

less=pd.DataFrame()
more=pd.DataFrame()

neg_cor=-0.85
pos_cor=0.85
   
for ticker in tickers[:100]:
    # print(ticker)
    # minimum=np.sort(df_corr['{}'.format(ticker)])[:5]
   
    less=pd.DataFrame(df_corr['{}'.format(ticker)][df_corr['{}'.format(ticker)]<neg_cor])

    if less.empty:
        print('===================',ticker)
        # less.index
    else:
        # print(ticker)
        
        print(less)
  

apple_corr=df_corr["AAPL"]      
    # more=pd.DataFrame(df_corr['{}'.format(ticker)][df_corr['{}'.format(ticker)]>pos_cor])

    # if more.empty:
    #     print('++++++++++')
    #     # less.index
    # else:
    #     print(ticker)
        
    #     print(more)

    # maximum=np.sort(df_corr['{}'.format(ticker)])[-6:-1]
    
    # pd.DataFrame(np.argsort(df_corr['{}'.format(ticker)])[:5])
        
    # if more.empty:
    #     more=pd.DataFrame(np.argsort(df_corr['{}'.format(ticker)])[-6:-1])
    # else:
    #     more=more.join(np.argsort(df_corr['{}'.format(ticker)])[-6:-1],how='outer')
 
# less=less.set_index(pd.Index([1,2,3,4,5]))
# more=more.set_index(pd.Index([1,2,3,4,5]))
        
plt.figure()
(sp500['AAPL']).plot()
(sp500['TGT']).plot()
(sp500['SYY']).plot()
# plt.yticks(np.arange(10,24,step=0.1))
plt.legend()
plt.show()

    
    

