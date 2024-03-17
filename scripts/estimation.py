import numpy as np
from tqdm import tqdm
import warnings


def gen_loc(loc_bound,grid_size,s=1,coor=np.zeros(2)):
  '''
  Generate a Cartesian grid of locations.
  loc_bound: location bound (can be a float or an array with four elements)
  s: scale parameter 
  grid_size: size of the grid
  coor: the coordinate of the center of the grid
  '''

  npos = grid_size**2
  if isinstance(loc_bound,list):
    a = np.linspace(loc_bound[0]/s+coor[0],loc_bound[1]/s+coor[0],grid_size)
    b = np.linspace(loc_bound[2]/s+coor[1],loc_bound[3]/s+coor[1],grid_size)
  else:
    a = np.linspace(-loc_bound/s+coor[0],loc_bound/s+coor[0],grid_size)
    b = np.linspace(-loc_bound/s+coor[1],loc_bound/s+coor[1],grid_size)
  loc = np.zeros((1,npos,2))
  for i in range(npos):
    loc[0,i,0] = a[int(i/grid_size)]
    loc[0,i,1] = b[int(i%grid_size)]
  return loc

def findlkd_no_constraint(num_position,dist,beta0,beta1,weight,bike_num,num_records,book_bike,book_index,book_num,all_period):
  '''
  Find the log-likelihood value under the observed data. The log-likelihood function is introduced in section 3.1 in the paper.
  '''
  choice_prob = np.zeros((num_position,bike_num+1,num_records))
  choice_prob[:,0,:] = 1/(1+np.sum(np.exp(beta0+beta1.reshape(-1,1,1)*dist),axis=2))
  choice_prob[:,1:,:] = np.exp(beta0+beta1.reshape(-1,1,1)*np.transpose(dist,(0,2,1)))/(1+np.sum(np.exp(beta0+beta1.reshape(-1,1,1)*dist),axis=2)).reshape(num_position,1,-1)
  weight = np.expand_dims(weight,1)
  return -book_num * np.log(np.sum((1-np.sum(weight*choice_prob[:,0,:],axis=0))*all_period)) + \
      np.sum(np.log(np.sum(weight*(choice_prob[:,book_bike+1,book_index]),axis=0)))

def caldist(loc,bike_loc,bike_num):
    '''
    Return a numpy array that contains the Euclidean distance between arrival locations and bikes at each time period. If a bike is 
    unavailable at some time, the distance from any arrival location to the bike at that time is recorded as infinity.
    '''
    loc = loc.reshape(-1,1,1,2)
    bike_loc = bike_loc.reshape(1,-1,bike_num,2)
    dist = np.sqrt(np.sum((loc-bike_loc)**2,axis=3))
    return dist

def findd(beta0,beta1,p_olds,f_old,N_oldy,dist,dist1,book_index,book_bike,all_period):
  '''
  Compute the partial derivative with respect to beta1 when beta1 is not given. When the partial derivative value is zero, we know beta1 
  is the maximizer in the current EM update. This is a helper function in using the binary search method to find beta1.
  '''
  p_j0tx = 1/(1+np.sum(np.exp(beta0+beta1*dist[:,book_index,:]),axis=2))
  p_j0t = 1/(1+np.sum(np.exp(beta0+beta1*dist),axis=2))
  return np.sum(np.sum(p_olds*(dist[:,book_index,book_bike]-p_j0tx*np.sum(dist1[:,book_index,:]*np.exp(beta0+beta1*dist[:,book_index,:]),axis=2)),axis=1) - \
        N_oldy*np.sum(f_old*p_j0t*np.sum(dist1*np.exp(beta0+beta1*dist),axis=2)*all_period,axis=1))

def findw_EM(cur_locnum,loc,beta0,bike_num,num_records,bike_loc,book_bike,num_booked,
             book_index,all_period,T,thres=5e-3, beta1 = -1, dist_old=None,prior_weight=None):
  
  '''
  Find the weigtht vector w using the EM algorithm. We start with a weight vector where all elements are the same and then iteratively
  update the weight vector w until the L-1 norm of the difference of two consecutive updates is less than some threshold.
  '''
  w = np.random.uniform(0,1,cur_locnum)
  w = w/np.sum(w)
  w = w.reshape(1,-1)
  beta1 = np.repeat(beta1,cur_locnum).reshape(-1,1)
  w_diff = np.inf
  if dist_old is None:
    dist_old = caldist(loc,bike_loc,bike_num)
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
    if prior_weight is not None:
      c = c+prior_weight
    w_star = np.reshape(c/np.sum(c),(1,-1))
    w = np.concatenate((w,w_star),axis=0)
    w_diff = np.sum(np.abs(w[-1]-w[-2]))
  return w[-1]

def findbetaw_EM(cur_locnum,loc,beta0,beta1,bike_num,num_records,bike_loc,book_bike,num_booked,book_index,all_period,T,thres=5e-3):
  '''
  Jointly estimate the weight vector w and the MNL model parameter beta1 using the EM algorithm. This is introudced in the Appendix as
  an extension of simply finding the weight vector w. beta0 is assumed to be given here. Otherwise, the non-identifiability issues will
  occur during the estimate.
  '''
  w = np.random.uniform(0,1,cur_locnum)
  w = w/np.sum(w)
  w = w.reshape(1,-1)
  w_diff = np.inf
  beta1 = np.array([beta1])
  dist_old = caldist(loc,bike_loc,bike_num)
  while (w_diff > thres):
    p_old = np.zeros((cur_locnum,bike_num+1,num_records))
    p_olds = np.zeros((cur_locnum,num_booked))
    f_old = np.zeros((cur_locnum,num_records))
    w_old = w[-1,:]  
    p_old[:,0,:] = 1/(1+np.sum(np.exp(beta0+beta1*dist_old),axis=2))
    p_old[:,1:,:] = np.exp(beta0+beta1*np.transpose(dist_old,(0,2,1)))/ \
      np.reshape((1+np.sum(np.exp(beta0+beta1*dist_old),axis=2)),(cur_locnum,1,-1))
    f_old = p_old[:,0,:]*np.reshape(w_old,(-1,1))/np.sum(np.sum(np.expand_dims(w_old,1)*p_old[:,0,:],axis=0)*all_period)
    p_olds = p_old[:,book_bike+1,book_index]*np.reshape(w_old,(-1,1))/np.reshape(np.sum(np.reshape(w_old,(-1,1))*p_old[:,book_bike+1,book_index],axis=0),(1,-1))
    p_oldsj0 = w_old*np.sum(p_old[:,0,:]*all_period,axis=1)/np.sum(w_old*np.sum(p_old[:,0,:]*all_period,axis=1))
    s_old = np.sum((1-np.sum(np.expand_dims(w_old,1)*(1/(1+np.sum(np.exp(beta0+beta1*dist_old),axis=2))),axis=0))*all_period)
    N_oldy = num_booked*(T-s_old)/s_old
    c = np.sum(p_olds,axis=1)+N_oldy*p_oldsj0
    w_star = np.reshape(c/np.sum(c),(1,-1))
    w = np.concatenate((w,w_star),axis=0)
    beta1_star = findbeta1em(p_olds,f_old,N_oldy,beta0,dist_old,book_index,book_bike,all_period,thres=1e-3)
    beta1 = beta1_star
    w_diff = np.sum(np.abs(w[-1]-w[-2]))
  return beta1, w[-1]

def findbeta1em(p_olds,f_old,N_oldy,beta0,dist,book_index,book_bike,all_period,minval=-20.0,maxval=-1e-8,thres=2*1e-4):
  '''
  Use binary Search to find the value of beta1. We naturally assume that beta1 is smaller than 0. We give an initial bound of beta1 and 
  repeatedly bisecting the interval until the absolute value of the partial derivative with respect to beta1 is less than a threshold.
  '''
  dist1 = dist.copy()
  dist1[dist1==np.inf] = 0
  beta1min = minval
  beta1max = maxval
  dmax = findd(beta0,beta1min,p_olds,f_old,N_oldy,dist,dist1,book_index,book_bike,all_period)
  dmin = findd(beta0,beta1max,p_olds,f_old,N_oldy,dist,dist1,book_index,book_bike,all_period)
  if (dmin>=0):
    beta1min = beta1max
    warnings.warn("warning:beta1 too large")
  if (dmax<=0):
    warnings.warn("warning:beta1 too small")
    beta1max = beta1min
  d = 1
  beta1 = beta1min
  while (np.any(abs(d)>thres)):
    beta1 = (beta1min+beta1max)/2
    d = findd(beta0,beta1,p_olds,f_old,N_oldy,dist,dist1,book_index,book_bike,all_period)
    if (dmin<0 and dmax>0):
      if (d<0):
        beta1max = beta1
      else:
        beta1min = beta1
    else:
      d = 0
  return beta1

def findprob0(dist_cur,genpos,beta0,beta1):
  '''
  Return the probability of leaving under the MNL model
  '''
  return 1/(1+np.sum(np.exp(beta0+beta1*dist_cur[genpos,:])))


def gen_sync(rand_seed,num_position,bike_num,lambd,grid_size,beta0=1,beta1_true=-1,T=50,loc_bound=5,split_data=False):  
  '''
  Generate an instance of synthetic data and return all relevant booking information.
  Arrival locations are generated uniformly within the range [-0.8*loc_bound, 0.8*loc_bound].
  rand_seed: random seed
  num_position: total number of arrival locations
  bike_num: total number of bikes
  grid_size: grid size of the candidate location
  beta0, beta1: MNL model parameter
  loc_bound: bound of the bike locations and arrival locations
  split_data: whether split the data into train/test set
  '''
  np.random.seed(seed=rand_seed)
  posx = np.random.uniform(-4*loc_bound/5,4*loc_bound/5,num_position)
  posy = np.random.uniform(-4*loc_bound/5,4*loc_bound/5,num_position)
  true_loc = np.stack((posx,posy),axis=1)

  position_weight = np.random.uniform(0,1,num_position)
  position_weight = position_weight / np.sum(position_weight)

  total_arr = np.random.poisson(lambd*T,size=1)[0]
  arr_time = np.sort(np.random.uniform(0,T,total_arr))
  tot_time = arr_time.copy()
  cand_loc = gen_loc(loc_bound,grid_size,s=1.25)


  book_bike = np.array([],dtype=int)
  x0 = np.random.uniform(0,loc_bound,bike_num)
  y0 = np.random.uniform(0,loc_bound,bike_num)
  dist = np.zeros((num_position,1,bike_num))
  bike_loc = np.zeros((1,bike_num,2))
  for i in range(num_position):
    dist[i,0,:] = np.sqrt((x0-posx[i])**2+(y0-posy[i])**2)
  bike_loc[0,:,:] = np.stack((x0,y0),axis=1)

  book_index_arr = np.array([],dtype=int)
  book_index = np.array([],dtype=int)
  ridgenpos = np.argmax(np.random.multinomial(1,position_weight,size=total_arr),axis=1)
  num_booked = 0
  duration = np.array([])
  book_time = np.array([])
  i = 0
  dist_i = 0
  finish_time = np.array([])
  k1 = 0
  xi = np.array([])
  yi = np.array([])
    
  while (i < tot_time.shape[0]):
    i = i + 1
    if tot_time[i-1] in finish_time:
      dist_i = dist_i + 1
      dist = np.hstack((dist,dist[:,dist_i-1,:].reshape(num_position,1,-1)))
      bike_loc = np.vstack((bike_loc,bike_loc[dist_i-1,:,:].reshape(1,bike_num,2)))
      bike_index = np.where((book_time + duration)==tot_time[i-1])[0][0]
      bike_loc[dist_i,int(book_bike[bike_index]),:] = [xi[bike_index], yi[bike_index]]
      dist[:,dist_i,int(book_bike[bike_index])] = np.sqrt((xi[bike_index]-posx)**2+(yi[bike_index]-posy)**2)
    else:
      k1 = k1 + 1
      prob_leave = findprob0(dist[:,dist_i,:],ridgenpos[k1-1],beta0,beta1_true)
      if not np.random.binomial(1,prob_leave):
        dist_i = dist_i + 1
        dist = np.hstack((dist,dist[:,dist_i-1,:].reshape(num_position,1,-1)))
        bike_loc = np.vstack((bike_loc,bike_loc[dist_i-1,:,:].reshape(1,bike_num,2)))
        
        book_index_arr = np.append(book_index_arr,i-1)
        book_time = np.append(book_time,tot_time[i-1])
        choiceprob = np.exp(beta1_true*dist[ridgenpos[k1-1],dist_i-1,:])/np.sum(np.exp(beta1_true*dist[ridgenpos[k1-1],dist_i-1,:]))
        book_bike = np.append(book_bike,np.argmax(np.random.multinomial(1,choiceprob,size=1)))
        book_index = np.append(book_index,dist_i-1)
        
        dist[:,dist_i,int(book_bike[num_booked])] = np.inf
        bike_loc[dist_i,int(book_bike[num_booked]),:] = [np.inf, np.inf]

        xi = np.append(xi,np.random.uniform(-loc_bound,loc_bound,1)[0])
        yi = np.append(yi,np.random.uniform(-loc_bound,loc_bound,1)[0])
        dist_to_dest = np.sqrt((xi[-1]-posx[ridgenpos[k1-1]])**2+(yi[-1]-posy[ridgenpos[k1-1]])**2)
        dur = np.max([dist_to_dest/18+ dist[ridgenpos[k1-1],dist_i-1,int(book_bike[num_booked])]/4+np.random.normal(0,0.1),0.05])
        duration = np.append(duration,dur)
        finish_time = np.append(finish_time,tot_time[i-1]+dur)
        tot_time = np.sort(np.append(tot_time,tot_time[i-1]+dur))
        book_index_arr[num_booked] = i-1
        num_booked = num_booked + 1
  book_finish_time = np.sort(np.concatenate((book_time,finish_time)))
  all_period = np.append(book_finish_time,[T])-np.append([0],book_finish_time)
  num_records = dist.shape[1]
  if split_data:
    time_portion = 0.8
    book_finish_time_train = book_finish_time[book_finish_time<time_portion*T]
    book_finish_time_test = book_finish_time[book_finish_time>=time_portion*T]
    train_period = np.append(book_finish_time_train,[time_portion*T])-np.append([0],book_finish_time_train)
    test_period = np.append(book_finish_time_test,[T])-np.append([time_portion*T],book_finish_time_test)
    num_records_train = train_period.shape[0]
    num_records_test = test_period.shape[0]
    num_booked_train = book_time[book_time<time_portion*T].shape[0]
    num_booked_test = num_booked-num_booked_train
    
    train_data = [bike_num,num_records_train,book_bike[:num_booked_train],book_index[:num_booked_train],dist[:,:num_records_train,:],
                  bike_loc[:num_records_train,:,:],train_period,num_booked_train,cand_loc,true_loc,position_weight]
    test_data = [bike_num,num_records_test,book_bike[num_booked_train:],book_index[num_booked_train:]-num_records_train+1,dist[:,(num_records_train-1):,:],
                  bike_loc[(num_records_train-1):,:,:],test_period,num_booked_test,cand_loc,true_loc,position_weight]
 
    return (train_data,test_data)
    
  return(bike_num,num_records,book_bike,book_index,dist,bike_loc,all_period,num_booked,cand_loc,true_loc,position_weight)

def gen_sync_in_grid(rand_seed,num_position,bike_num,lambd,grid_size,beta0=1,beta1_true=-1,T=50,loc_bound=5):
  '''
  Generate an instance of synthetic data and return all relevant booking information.
  Arrival locations are generated from a candidate grid where the size of the grid is grid_size.
  '''
  np.random.seed(seed=rand_seed)
  true_pos_ind = np.random.choice(grid_size**2,num_position,replace=False)
  position_weight = np.random.uniform(0,1,num_position)
  position_weight = position_weight / np.sum(position_weight)

  total_arr = np.random.poisson(lambd*T,size=1)[0]
  arr_time = np.sort(np.random.uniform(0,T,total_arr))
  tot_time = arr_time.copy()
  cand_loc = gen_loc(loc_bound,grid_size,s=1.25)
  posx = cand_loc[0,true_pos_ind,0]
  posy = cand_loc[0,true_pos_ind,1]
  true_loc = np.stack((posx,posy),axis=1)

  book_bike = np.array([],dtype=int)
  x0 = np.random.uniform(0,loc_bound,bike_num)
  y0 = np.random.uniform(0,loc_bound,bike_num)
  dist = np.zeros((num_position,1,bike_num))
  bike_loc = np.zeros((1,bike_num,2))
  for i in range(num_position):
    dist[i,0,:] = np.sqrt((x0-posx[i])**2+(y0-posy[i])**2)
  bike_loc[0,:,:] = np.stack((x0,y0),axis=1)
  book_index_arr = np.array([],dtype=int)
  book_index = np.array([],dtype=int)
  ridgenpos = np.argmax(np.random.multinomial(1,position_weight,size=total_arr),axis=1)
  num_booked = 0
  duration = np.array([])
  book_time = np.array([])
  i = 0
  dist_i = 0
  finish_time = np.array([])
  k1 = 0
  xi = np.array([])
  yi = np.array([])

  while (i < tot_time.shape[0]):
    i = i + 1
    if tot_time[i-1] in finish_time:
      dist_i = dist_i + 1
      dist = np.hstack((dist,dist[:,dist_i-1,:].reshape(num_position,1,-1)))
      bike_loc = np.vstack((bike_loc,bike_loc[dist_i-1,:,:].reshape(1,bike_num,2)))
      bike_index = np.where((book_time + duration)==tot_time[i-1])[0][0]
      bike_loc[dist_i,int(book_bike[bike_index]),:] = [xi[bike_index], yi[bike_index]]
      dist[:,dist_i,int(book_bike[bike_index])] = np.sqrt((xi[bike_index]-posx)**2+(yi[bike_index]-posy)**2)
    else:
      k1 = k1 + 1
      prob_leave = 1/(1+np.sum(np.exp(beta0+beta1_true*dist[ridgenpos[k1-1],dist_i,:])))
      if not np.random.binomial(1,prob_leave):
        dist_i = dist_i + 1
        dist = np.hstack((dist,dist[:,dist_i-1,:].reshape(num_position,1,-1)))
        bike_loc = np.vstack((bike_loc,bike_loc[dist_i-1,:,:].reshape(1,bike_num,2)))
        
        book_index_arr = np.append(book_index_arr,i-1)
        book_time = np.append(book_time,tot_time[i-1])
        choiceprob = np.exp(beta1_true*dist[ridgenpos[k1-1],dist_i-1,:])/np.sum(np.exp(beta1_true*dist[ridgenpos[k1-1],dist_i-1,:]))
        book_bike = np.append(book_bike,np.argmax(np.random.multinomial(1,choiceprob,size=1)))
        book_index = np.append(book_index,dist_i-1)
        
        dist[:,dist_i,int(book_bike[num_booked])] = np.inf
        bike_loc[dist_i,int(book_bike[num_booked]),:] = [np.inf, np.inf]

        xi = np.append(xi,np.random.uniform(-loc_bound,loc_bound,1)[0])
        yi = np.append(yi,np.random.uniform(-loc_bound,loc_bound,1)[0])
        dist_to_dest = np.sqrt((xi[-1]-posx[ridgenpos[k1-1]])**2+(yi[-1]-posy[ridgenpos[k1-1]])**2)
        dur = np.max([dist_to_dest/18+ dist[ridgenpos[k1-1],dist_i-1,int(book_bike[num_booked])]/4+np.random.normal(0,0.1),0.05])
        duration = np.append(duration,dur)
        finish_time = np.append(finish_time,tot_time[i-1]+dur)
        tot_time = np.sort(np.append(tot_time,tot_time[i-1]+dur))
        book_index_arr[num_booked] = i-1
        num_booked = num_booked + 1

  book_finish_time = np.sort(np.concatenate((book_time,finish_time)))
  all_period = np.append(book_finish_time,[T])-np.append([0],book_finish_time)
  num_records = dist.shape[1]
  return true_pos_ind,bike_num,num_records,book_bike,book_index,dist,bike_loc,all_period,num_booked,cand_loc,true_loc,position_weight

def g(w,choice_prob,book_bike,book_index):
    g = np.sum(np.log(np.sum(w.reshape(-1,1)*(choice_prob[:,book_bike+1,book_index]),axis=0)))
    return g

def h(w,choice_prob,all_period,num_booked):
    h = num_booked * np.log(np.sum((1-np.sum(w.reshape(-1,1)*choice_prob[:,0,:],axis=0))*all_period))
    return h

def find_grad_g(w,choice_prob,book_bike,book_index):
    grad_g = np.sum(choice_prob[:,book_bike+1,book_index]/np.sum(w.reshape(-1,1)*(choice_prob[:,book_bike+1,book_index]),axis=0),axis=1)
    return grad_g
