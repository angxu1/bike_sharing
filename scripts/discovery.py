import numpy as np

def calgrad(book_num,s,num_position,beta0,beta1,choice_prob,all_period,weight,dist_new,bike_num,num_records,book_bike,book_index):
  '''
  Compute the partial derivative of the new location under the current weight. Larger partial derivative indicates a better potentiality
  to improve the log-likelihood function value in the EM iteration.
  new_loc: (n,2), n is num of new candidate locatios
  dist_new: distance matrix of cand locs
  prob_new: prob matrix for new cand locs
  '''
  weight = weight.reshape(-1,1)
  prob_new = np.zeros((num_position,bike_num+1,num_records))
  prob_new[:,0,:] = 1/(1+np.sum(np.exp(beta0+beta1*dist_new),axis=2))
  prob_new[:,1:,:] = np.exp(beta0+beta1*np.transpose(dist_new,(0,2,1)))/(1+np.sum(np.exp(beta0+beta1*dist_new),axis=2)).reshape(num_position,1,-1)
  return book_num/s*np.sum(prob_new[:,0,:]*all_period,axis=1) + \
    np.sum(prob_new[:,book_bike+1,book_index]/np.sum(weight*choice_prob[:,book_bike+1,book_index],axis=0).reshape(1,-1),axis=1)

def findchoice_prob(cur_locnum,cur_dist,beta0,beta1,num_records,bike_num):
  '''
  Return an array that records the choice probabilitiy under the MNL model given the arrival locations and the bike pattern.
  We assume that the model parameter beta0 and beta1 are known.
  '''
  choice_prob = np.zeros((cur_locnum,bike_num+1,num_records))
  choice_prob[:,0,:] = 1/(1+np.sum(np.exp(beta0+beta1*cur_dist),axis=2))
  choice_prob[:,1:,:] = np.exp(beta0+beta1*np.transpose(cur_dist,(0,2,1))) / \
    (1+np.sum(np.exp(beta0+beta1*cur_dist),axis=2)).reshape(cur_locnum,1,-1)
  return choice_prob

def sel_loc_max(grad,grid_size):
    '''
    Return a location set that contains all locations that has larger partial derivative than all its neighbors.
    grad: 2d array that records the partial derivative at each location on the grid
    grid_size: size of the grid
    '''
    grad = grad.reshape(grid_size,grid_size)
    grad_ext = np.zeros((grid_size+2,grid_size+2))
    grad_ext[1:grid_size+1,1:grid_size+1] = grad
    loc_set = np.array([],dtype=int)
    for i in range(grid_size):
        for j in range(grid_size):
            if np.all(grad_ext[i+1,j+1]-grad_ext[i:(i+3),j:(j+3)] >= 0):
                loc_set = np.append(loc_set,i*grid_size+j)
    return loc_set