import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from scripts.discovery import findchoice_prob
from shapely.geometry import Point
from datetime import datetime, date, time

def getgeodist(lng1,lng2,lat1,lat2):
  '''
  Compute the geometric distance between two locations given the corresponding longitudes and latitudes
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
  Compute the geometric distance between two locations given the corresponding longitudes and latitudes
  lng1,lng2: the longitude of location 1 and location 2 
  lat1,lat2: the latitude of location 1 and location 2 
  '''
  if (lng1==np.inf) or (lng2==np.inf) or (lat1==np.inf) or (lat2==np.inf):
    return np.inf
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

def caldist_geo(loc,bike_loc):
  loc = loc.reshape(-1,2)
  bike_num = bike_loc.shape[1]
  num_position = loc.shape[0]
  num_records = bike_loc.shape[0]
  dist = np.zeros((num_position,num_records,bike_num))
  for j in range(bike_num):
    if bike_loc[0,j,0]==np.inf:
      dist[:,0,j] = np.inf
    else:
      dist[:,0,j] = getgeodist(bike_loc[0,j,0],loc[:,0],bike_loc[0,j,1],loc[:,1])
  for i in range(1,num_records):
    dist[:,i,:] = dist[:,i-1,:]
    diff_loc_ind1 = np.where(bike_loc[i,:,0]!=bike_loc[i-1,:,0])[0]
    diff_loc_ind2 = np.where(bike_loc[i,:,1]!=bike_loc[i-1,:,1])[0]
    diff_loc_ind = np.intersect1d(diff_loc_ind1,diff_loc_ind2)
    for ind in diff_loc_ind:
      if (bike_loc[i,ind,0]!=np.inf) and (bike_loc[i,ind,1]!=np.inf):
        dist[:,i,ind] = getgeodist(loc[:,0],bike_loc[i,ind,0],loc[:,1],bike_loc[i,ind,1])
      else:
        dist[:,i,ind] = np.inf
        
  return dist


def findw_EM(cur_locnum,loc,bike_num,num_records,book_bike,num_booked,bike_loc,
             book_index,all_period,T,thres=1e-2, beta0=1,beta1=-5):
  '''
  Find the weigtht vector w using the EM algorithm. We start with a weight vector where all elements are the same and then iteratively
  update the weigth vector w until the L-1 norm of the difference of two consecutive updates is less than some threshold.
  '''
  w = np.random.uniform(0,1,cur_locnum)
  w = w/np.sum(w)
  w = w.reshape(1,-1)
  beta1 = np.repeat(beta1,cur_locnum).reshape(-1,1)
  w_diff = np.inf
  dist_old = caldist_geo(loc.reshape(-1,2),bike_loc)
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



def findbetaw1_EM(cur_locnum,loc,beta0,beta1,bike_num,num_records,bike_loc,book_bike,num_booked,book_index,all_period,T,thres=5e-3,prior_weight=None):
  '''
  Jointly estimate the weight vector w and the MNL model parameter beta1 using the EM algorithm. To make the algorithm more efficient, we search five values within
  the neighborhood of the current estimation.
  '''
  w = np.random.uniform(0,1,cur_locnum)
  w = w/np.sum(w)
  w = w.reshape(1,-1)
  w_diff = np.inf
  dist_old = caldist_geo(loc,bike_loc)
  while (w_diff > thres): 
    p_old = np.zeros((cur_locnum,bike_num+1,num_records))
    p_olds = np.zeros((cur_locnum,num_booked))
    w_old = w[-1,:]  
    p_old[:,0,:] = 1/(1+np.sum(np.exp(beta0+beta1*dist_old),axis=2))
    p_old[:,1:,:] = np.exp(beta0+beta1*np.transpose(dist_old,(0,2,1)))/ \
      np.reshape((1+np.sum(np.exp(beta0+beta1*dist_old),axis=2)),(cur_locnum,1,-1))
    p_olds = p_old[:,book_bike+1,book_index]*np.reshape(w_old,(-1,1))/np.reshape(np.sum(np.reshape(w_old,(-1,1))*p_old[:,book_bike+1,book_index],axis=0),(1,-1))
    p_oldsj0 = w_old*np.sum(p_old[:,0,:]*all_period,axis=1)/np.sum(w_old*np.sum(p_old[:,0,:]*all_period,axis=1))
    s_old = np.sum((1-np.sum(np.expand_dims(w_old,1)*(1/(1+np.sum(np.exp(beta0+beta1*dist_old),axis=2))),axis=0))*all_period)
    N_oldy = num_booked*(T-s_old)/s_old
    c = np.sum(p_olds,axis=1)+N_oldy*p_oldsj0
    if prior_weight is not None:
      c = c+prior_weight
    w_star = np.reshape(c/np.sum(c),(1,-1))
    w = np.concatenate((w,w_star),axis=0)
    
    beta1_cand = [beta1-0.2,beta1-0.1,beta1,beta1+0.1,beta1+0.2]
    beta1_star = beta1_cand[np.argmax([findlkd_beta(beta0,b,p_old,p_olds,dist_old,s_old,book_index,book_bike,w_old,bike_num,num_records,num_booked,
                                                    all_period,cur_locnum) for b in beta1_cand])]
    beta1 = beta1_star
    w_diff = np.sum(np.abs(w[-1]-w[-2]))
    #print(w_diff,beta1)
  return beta1, w[-1]

def findlkd_beta(beta0,beta1_new,p_old,p_olds,dist_old,s,book_index,book_bike,w_old,bike_num,num_records,num_booked,all_period,cur_locnum):
  choice_prob_new = findchoice_prob(cur_locnum,dist_old,beta0,beta1_new,num_records,bike_num)
  beta1_em_lkd = np.sum(np.sum(p_olds*np.log(choice_prob_new[:,book_bike+1,book_index]),axis=1)+
                        num_booked/s*np.sum(p_old[:,0,:]*w_old.reshape(-1,1)*np.log(choice_prob_new[:,0,:])*all_period,axis=1))
  #print(beta1_new,beta1_em_lkd)
  return beta1_em_lkd


def get_tourist_attractions(api_key, city="Seattle", type="tourist_attraction"):
  endpoint_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
  search_query = f"{type} in {city}"

  params = {
      'query': search_query,
      'key': api_key
  }

  response = requests.get(endpoint_url, params=params).json()

  if response['status'] == 'OK':
      attractions = []
      for result in response['results']:
          attractions.append({
              'name': result['name'],
              'location': result['geometry']['location']
          })
      return attractions
  else:
      print(f"Error: {response['status']}")
      return None

def get_metro_stations(api_key, city="Seattle", type="subway_station"):
  endpoint_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
  search_query = f"{type} in {city}"

  params = {
      'query': search_query,
      'key': api_key
  }

  response = requests.get(endpoint_url, params=params).json()

  if response['status'] == 'OK':
      stations = []
      for result in response['results']:
          stations.append({
              'name': result['name'],
              'location': result['geometry']['location']
          })
      return stations
  else:
      print(f"Error: {response['status']}")
      return None

def time_in_hours(time_str):
  # Parse the time string (including milliseconds)
  time_obj = datetime.strptime(time_str, '%H:%M:%S.%f')
  # Convert to hours (including milliseconds)
  hours = time_obj.hour + time_obj.minute / 60 + time_obj.second / 3600 + time_obj.microsecond / 3600000000
  return hours

def is_time_in_range(time_str,start_hours,end_hours):
  time_hours = time_in_hours(time_str)
  return start_hours <= time_hours <= end_hours

def change_df_time_format(df):
  df = df.sort_values(by=['time'])
  initial_time = np.datetime64('2019-08-01 00:00:00')
  all_time = np.array([np.datetime64(date_str) for date_str in df['time']])
  all_time = np.append([initial_time], all_time)
  book_finish_time = (all_time[1:] - initial_time) / np.timedelta64(1, 's')
  book_finish_time
  df['time'] = book_finish_time
  return df

def create_bike_loc(df, lng0, lat0,total_day,initial_time = np.datetime64('2019-07-01 00:00:00')):
  sec_day = 86400
  total_period = total_day*sec_day
  unique_times = np.sort(df['time'].unique())
  all_time = np.sort(df['time'].unique())
  all_time = np.array([np.datetime64(date_str) for date_str in all_time])
  all_time = np.append([initial_time], all_time)
  book_finish_time = (all_time[1:] - initial_time) / np.timedelta64(1, 's')
  all_period = np.append(book_finish_time,total_period)-np.append([0],book_finish_time)
  
  book_time = df[df['state']=='rent_start']['time']
  book_time = np.sort(book_time)
  book_time = np.array([np.datetime64(date_str) for date_str in book_time])
  book_time = (book_time-initial_time)/ np.timedelta64(1, 's')

  book_index  = np.array([np.where(book_finish_time == element)[0][-1] for element in book_time]) # Find the index of the booking time in the book_finish_time array
  df1 = df[df['state']=='rent_start'].sort_values(by=['time'])
  bike_index = df1['bike_id'].astype(int) # Find the index of the bike in the bike_loc array
  bike_index = np.array(bike_index)-1
  bike_ids = df['bike_id'].unique()
  bike_num = len(bike_ids)
  num_records = len(all_time)
  bike_loc = np.zeros((num_records, bike_num, 2)) 
  print(num_records,unique_times.shape,all_time.shape)
  for i in range(num_records):
      for j in range(bike_num):
          bike_loc[i,j] = [lng0[j], lat0[j]]


  df_sorted = df.sort_values(by=['bike_id', 'time'])

  for _, row in tqdm(df_sorted.iterrows()):
      bike_id = row['bike_id'] - 1
      time = row['time']
      state = row['state']
      lng, lat = row['lng'], row['lat']
      time_index = np.where(unique_times == time)[0][0]

      if state == 'rent_finish':
          bike_loc[time_index+1, bike_id] = [lng, lat]
      elif state == 'rent_start':
          bike_loc[time_index+1, bike_id] = [np.inf, np.inf]

      if time_index + 2 < len(all_time):
          bike_loc[time_index + 2:, bike_id] = bike_loc[time_index+1, bike_id]

  return bike_loc,all_period,book_time,book_finish_time,book_index,bike_index

# For each day, initialize the bike pattern as the most recent record 

def extract_period_data(bike_loc,all_period,book_time,book_finish_time,book_index,bike_index,total_day,time_interval):
  # time_interval: [start hour in a day,end hour in a day)
  sec_day = 86400
  sec_hour = 3600
  hour_day = 24
  time_diff = time_interval[1]-time_interval[0]
  total_period = total_day*sec_hour*time_diff
  #df_coor = df_coor
  def check_valid_time(time):
      time_hour = (time/sec_hour)%hour_day
      return ((time_hour>=time_interval[0]) & (time_hour<time_interval[1]))
  
  alt_time = np.arange(0,total_day)*sec_day+sec_hour*time_interval[0] # Alternative time at the beginning of each time interval
  # Add a df to reserve lng,lat 
  
  #df_coor = df_coor[check_valid_time(t) for t in df_coor['time']]
  # Compute the index of the alternative time in the book_finish_time array
  alt_index = np.zeros(total_day,dtype=int)
  for i in range(total_day):
      book_finish_time_pre = book_finish_time[book_finish_time<alt_time[i]]
      alt_index[i] = len(book_finish_time_pre) # Find the index of the alternative time in the book_finish_time array
      
  # Add the alternative time at the beginning of each time interval
  for i in range(total_day):
      bike_loc = np.insert(bike_loc,alt_index[i]+1+i,axis=0,values=bike_loc[alt_index[i]+i,:,:])    
      book_finish_time = np.insert(book_finish_time,alt_index[i]+i,axis=0,values=alt_time[i])

  # Compute the valid time
  valid_time = np.array([check_valid_time(time) for time in book_finish_time])
  book_valid_time = np.array([check_valid_time(time) for time in book_time])
  
  book_time_in_interval = book_time[book_valid_time]
  book_time_in_interval = np.array([t%sec_day+int(t/sec_day)*sec_hour*time_diff for t in book_time_in_interval])
  book_time_in_interval -= time_interval[0]*sec_hour
  book_finish_time_in_interval = book_finish_time[valid_time]
  book_finish_time_in_interval = np.array([t%sec_day+int(t/sec_day)*sec_hour*time_diff for t in book_finish_time_in_interval])      
  book_finish_time_in_interval -= time_interval[0]*sec_hour
  #print(np.all(np.isin(book_time_in_interval,book_finish_time_in_interval)))
  bike_loc_in_interval = bike_loc[np.append([False],valid_time),:,:]
  bike_loc_in_interval = np.insert(bike_loc_in_interval,0,axis=0,values=bike_loc_in_interval[0,:,:])
  book_index_in_interval = np.array([np.where(book_finish_time_in_interval == element)[0][-1] for element in book_time_in_interval])
  bike_index_in_interval = bike_index[book_valid_time]    
  all_period_in_interval = np.append(book_finish_time_in_interval,total_period)-np.append([0],book_finish_time_in_interval)
  
  return bike_loc_in_interval,book_time_in_interval,book_finish_time_in_interval, \
      book_index_in_interval,bike_index_in_interval,all_period_in_interval
      
def find_bike_index(bike_loc,polygon_sets,num_block):
  num_rec_test, num_bike_test,_ = bike_loc.shape
  bike_poly_index = -np.ones((num_rec_test,num_bike_test),dtype=int)
  for i in tqdm(range(num_rec_test)):
      if i==0:
          for j in range(num_bike_test):
              for k in range(num_block):
                  if polygon_sets[k].contains(Point(bike_loc[i,j,0],bike_loc[i,j,1])):
                      bike_poly_index[i,j] = k            
      else:
          diff_bike_ind = 0
          bike_poly_index[i] = bike_poly_index[i-1]
          for j in range(num_bike_test):
              if (bike_loc[i,j,0] != bike_loc[i-1,j,0]) or (bike_loc[i,j,1] != bike_loc[i-1,j,1]):
                  diff_bike_ind = j
                  bike_poly_index[i,diff_bike_ind] = -1
                  for k in range(num_block):
                      if polygon_sets[k].contains(Point(bike_loc[i,diff_bike_ind,0],bike_loc[i,diff_bike_ind,1])):
                          bike_poly_index[i,diff_bike_ind] = k
                          break   
  return bike_poly_index    
      
  
def filter_bikes_in_region(df, lng1, lng2, lat1, lat2):
  # Filter for bookings within the region
  df = df[(df['lng'] >= lng1) & (df['lng'] <= lng2) & (df['lat'] >= lat1) & (df['lat'] <= lat2)]
  return df

def process_bike_data(df,enforce_seq=True):
  # Task 1: Reset bike_id
  # Task 2: Extract initial bike position
  # Task 3: Enforce rent_start/rent_finish sequence
  def check_sequence(group):
      valid_rows = []
      last_rent_start_index = None

      for index, row in group.iterrows():
          if row['state'] == 'rent_start':
              last_rent_start_index = index
          elif row['state'] == 'rent_finish':
              if last_rent_start_index is not None:
                  valid_rows.append(group.loc[last_rent_start_index])
                  last_rent_start_index = None
              valid_rows.append(row)

      return pd.DataFrame(valid_rows)
  if enforce_seq:
      df = df.sort_values(by=['bike_id', 'time'])
      df = df.groupby('bike_id').apply(check_sequence).reset_index(drop=True)
  
  initial_positions = df.sort_values(by=['bike_id', 'time']).drop_duplicates(subset='bike_id', keep='first')
  initial_positions = initial_positions[['bike_id', 'lng', 'lat','state']]
  # if state is rent_finish, then set lng and lat to be np.inf
  initial_positions.loc[initial_positions['state']=='rent_finish','lng'] = np.inf
  initial_positions.loc[initial_positions['state']=='rent_finish','lat'] = np.inf
  
  return df, initial_positions

def reset_bike_index(df):
  unique_bike_ids = df['bike_id'].unique()
  bike_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_bike_ids, start=1)}
  df['bike_id'] = df['bike_id'].map(bike_id_map)
  return df

def whether_in_land(gdf_naive,loc_cor,num_block):
    in_land = False
    for j in range(1,num_block):
        if gdf_naive['geometry'][j].contains(Point(loc_cor[0],loc_cor[1])):
            in_land = True
    return in_land
