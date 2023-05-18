import numpy as np
import pandas as pd
from datetime import datetime, date, time

def getgeodist(lng1,lng2,lat1,lat2):
  '''
  Compute the geometrical distance between two locations given the corresponding longitudes and latitudes
  lng1,lng2: the longitude of location 1 and location 2 
  lat1,lat2: the latitude of location 1 and location 2 
  '''
  lng1 = lng1/180*np.pi
  lng2 = lng2/180*np.pi
  lat1 = lat1/180*np.pi
  lat2 = lat2/180*np.pi
  a = np.sin((lat2-lat1) / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2-lng1)/2)**2
  c = 2 * 6373 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
  return c

def getgeodist_arr(lng1,lng2,lat1,lat2):
  '''
  Compute the geometrical distance between two locations given the corresponding longitudes and latitudes
  lng1,lng2: the longitude of location 1 and location 2 
  lat1,lat2: the latitude of location 1 and location 2 
  '''
  lng1 = lng1/180*np.pi
  lng2 = lng2/180*np.pi
  lat1 = lat1/180*np.pi
  lat2 = lat2/180*np.pi
  lng1 = lng1.reshape(1,-1)
  lng2 = lng2.reshape(-1,1)
  lat1 = lat1.reshape(1,-1)
  lat2 = lat2.reshape(-1,1)
  a = np.sin((lat2-lat1) / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2-lng1)/2)**2
  c = 2 * 6373 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
  return c

# 1. Reset bike_id
# 2. Extract data for the initial bike position information
# 3. Remove rows with rent_finish before rent_start
def recon_df(df):
    '''
    This function performs the following three tasks on the input dataframe:
    1. Reset bike_id
    2. Extract data for the initial bike position information
    3. Remove rows with rent_finish before rent_start
    '''
    df = df.reset_index(drop=True)
    reind = 0
    rent_fin = True
    delrow = np.array([])
    tempdel = 0
    for i in range(df.shape[0]-1):
        if df['state'][i] == "rent_start":
            if rent_fin:
                rent_fin = False
                tempdel = i
            else:
                delrow = np.append(delrow, i)
        elif df['state'][i] == "rent_finish":
            if not rent_fin:
                rent_fin = True
            else:
                delrow = np.append(delrow, i)
        cur_id = df.at[i,'bike_id']
        df.at[i,'bike_id'] = reind
        if df['bike_id'][i+1] != cur_id:
            reind = reind + 1
            if not rent_fin:
                delrow = np.append(delrow, tempdel)
            rent_fin = True
    df.at[df.shape[0]-1,'bike_id'] = reind
    i = df.shape[0]-1
    if df['state'][i] == "rent_start":
        delrow = np.append(delrow, i)
    elif df['state'][i] == "rent_finish":
        if not rent_fin:
            rent_fin = True
        else:
            delrow = np.append(delrow, i)
    df = df.drop(delrow)
    df_init = df.drop_duplicates(subset=['bike_id'])
    return df,df_init

def rent_record(df):
    df_rent = df.sort_values(by='time')
    df_rent = df_rent[df_rent['state']!='available'].reset_index(drop=True)
    return df_rent

def caldist2(loc,num_position,df_rent,num_records,bike_num,x0,y0):
  loc = loc.reshape(1,-1,2)
  dist = np.zeros((num_position,num_records,bike_num))
  dist[:,0,:] = getgeodist_arr(x0,loc[0,:,0],y0,loc[0,:,1])
  for i in range(1,num_records):
    dist[:,i,:] = dist[:,i-1,:]
    cur_rec = df_rent.loc[i-1,:]
    if cur_rec['state']=='rent_start':
      dist[:,i,cur_rec['bike_id']] = np.inf
    else:
      dist[:,i,cur_rec['bike_id']] = getgeodist(cur_rec['lng'],loc[0,:,0],cur_rec['lat'],loc[0,:,1])
  return dist

def findw_EM(cur_locnum,loc,bike_num,num_records,book_bike,num_booked,
             book_index,all_period,T,df_rent,x0,y0,thres=1e-2, beta0=1,beta1=-5):
  '''
  Find the weigtht vector w using the EM algorithm in the Seattle data. We start with a weight vector where all elements are the same and then iteratively
  update the weigth vector w until the L-1 norm of the difference of two consecutive updates is less than some threshold.
  '''
  w = np.random.uniform(0,1,cur_locnum)
  w = w/np.sum(w)
  w = w.reshape(1,-1)
  beta1 = np.repeat(beta1,cur_locnum).reshape(-1,1)
  w_diff = np.inf
  dist_old = caldist2(loc,cur_locnum,df_rent,num_records,bike_num,x0,y0)
  while (w_diff > thres):
    p_old = np.zeros((cur_locnum,bike_num+1,num_records))
    p_olds = np.zeros((cur_locnum,num_booked))
    w_old = w[-1,:]
    p_old[:,0,:] = 1/(1+np.sum(np.exp(beta0+np.reshape(beta1,(-1,1,1))*dist_old),axis=2))
    p_old[:,1:,:] = np.exp(beta0+np.reshape(beta1,(-1,1,1))*np.transpose(dist_old,(0,2,1)))/ \
      np.reshape((1+np.sum(np.exp(beta0+np.reshape(beta1,(-1,1,1))*dist_old),axis=2)),(cur_locnum,1,-1))
    p_olds = p_old[:,book_bike+1,book_index]*np.reshape(w_old,(-1,1))/np.reshape(np.sum(np.reshape(w_old,(-1,1))*p_old[:,book_bike+1,book_index],axis=0),(1,-1))
    p_oldsj0 = w_old*np.sum(p_old[:,0,:]*all_period,axis=1)/np.sum(w_old*np.sum(p_old[:,0,:]*all_period,axis=1))
    s_old = np.sum((1-np.sum(np.expand_dims(w_old,1)*(1/(1+np.sum(np.exp(beta0+np.reshape(beta1,(-1,1,1))*dist_old),axis=2))),axis=0))*all_period)
    N_oldy = num_booked*(T-s_old)/s_old
    c = np.sum(p_olds,axis=1)+N_oldy*p_oldsj0
    w_star = np.reshape(c/np.sum(c),(1,-1))
    w = np.concatenate((w,w_star),axis=0)
    w_diff = np.sum(np.abs(w[-1]-w[-2]))
  return w[-1]




