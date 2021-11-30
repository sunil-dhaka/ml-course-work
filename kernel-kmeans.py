import numpy as np
import numpy.linalg as lin
import numpy.random as rnd
from matplotlib import pyplot as plt


def getFigure( sizex = 7, sizey = 7 ):
    fig = plt.figure( figsize = (sizex, sizey) )
    return fig

def plot2D( X, fig, color = 'r', marker = '+', size = 100, empty = False ):
    plt.figure( fig.number )
    if empty:
        plt.scatter( X[:,0], X[:,1], s = size, facecolors = 'none', edgecolors = color, marker = marker  )
    else:
        plt.scatter( X[:,0], X[:,1], s = size, c = color, marker = marker )


def genCrescentData( d, n, mu, r, flipped = False ):
    X = np.vstack( (np.cos( np.linspace( 0, np.pi, n ) ), np.sin( np.linspace( 0, np.pi, n ) ) ) ).T
    if flipped:
        X[:,1] = -np.abs( X[:,1] )
    else:
        X[:,1] = np.abs( X[:,1] )
    X = (X * r) + mu
    return X

def genSphericalData( d, n, mu, r ):
    X = rnd.normal( 0, 1, (n, d) )
    norms = lin.norm( X, axis = 1 )
    X = X / norms[:, np.newaxis]
    X = (X * r) + mu
    return X

d = 2
n = 200

mu1 = np.array( [0,0] )
mu2 = np.array( [0,1] )
mu3 = np.array( [0,0] )
mu4 = np.array( [-3,5] )
mu5 = np.array( [3,5] )

tmp1 = genCrescentData( d, n, mu1, 1 )
tmp2 = genCrescentData( d, n, mu2, 5, flipped = True )
tmp3 = genSphericalData( d, n, mu3, 10 )
tmp4 = genSphericalData( d, n, mu4, 1 )
tmp5 = genSphericalData( d, n, mu5, 1 )
X = np.vstack( (tmp1, tmp2, tmp3, tmp4, tmp5) )

'''
func to plot 2D data with cluster IDs
Input:
data_2d: Data points in 2D :: If higher dim then can give PCA(2) applied data
labels: cluster IDs array
n_cluster: no of clusters; default = 4
Output:
plots a 2D scatter map with legend
'''
def plot_clusters(data_2d,labels,n_cluster=4,iter=20):
    # filter rows of original data
    # for label in labels:
    fig = plt.figure( figsize = (8, 8) )
    plt.figure(fig.number)
    for cluster in range(n_cluster):
        label0 = data_2d[labels == cluster]
        plt.scatter(label0[:,0] , label0[:,1],label=cluster)
    plt.legend(fontsize='small')
    plt.savefig(f'{n_cluster}_{iter}.png')
    plt.show()

from time import time
# set this variable to change lambda in guassian kernel
LAMBDA=1
def pair_kernel_dist(x,y):
    return np.exp(-LAMBDA * np.sum((x-y)**2))
    
def pair_kernel_dist_total(x,cluster_points):
    # math is bit involving given in markdowns
    # tic=time()
    n_points_=len(cluster_points)
    return_dist=0
    for i,point1 in enumerate(cluster_points):
        return_dist-=(2/n_points_)*pair_kernel_dist(x,point1)
        
    # toc=time()
    # print(toc-tic)
    return return_dist
def kernel_dist(X,centroid):
    # tic=time()
    n_points_=len(centroid)
    centroid_points_dist=n_points_
    for i in range(n_points_):
        for j in range(n_points_):
            # this one is 2 times faster than the below one
            if j>i:
                centroid_points_dist+=2*pair_kernel_dist(centroid[i,],centroid[j,])
            # below one is inefficient
            # if i!=j:
            #     centroid_points_dist+=pair_kernel_dist(centroid[i,],centroid[j,])
    centroid_points_dist=centroid_points_dist/(n_points_**2)
    # toc=time()
    # print(toc-tic)
    
    return_array=np.empty(X.shape[0])
    for i,x in enumerate(X):
        return_array[i]=pair_kernel_dist_total(x,centroid)+centroid_points_dist
    
    return return_array
def k(X,centroid):
    # exact formula is given in kernel kmeans slide 
    return 2*(1-(np.exp(-LAMBDA * np.sum((X-centroid)**2,axis=1))))

'''
k menas clustering algo implementation
Input:
X: data points
n_cluster: no of clusters; default = 4
init_style: initial cluster choosing style; default = k-menas
    - k-means: to choose initial clusters randomly
    - k-menas++: to choose initial clusters usefully
Output:
Cluster labels for data points
''' 
def get_kernel_KMeans_clusters(X,n_cluster=4,init_style='k-means',n_iter=2):

    # to reproduce results
    # np.random.seed(1)

    n_points=X.shape[0]
    n_dim=X.shape[1]

    '''
    Get Inital Centroids
    '''
    # cluster centers are chosen to be K of the data points themselves #
    if init_style=='k-means':

        # this method simply chooses random n_cluster points from permuatated index
        init_centroids_index=rnd.permutation(n_points)[:n_cluster]
        init_centroids=X[init_centroids_index]
        # another way is to create k-many random centroids that are not data points
        #TODO: implement this one also with if-else cond
    elif init_style=='k-means++':
        # in this we choose centroids that are more representative of the sample points
        # empty array and list
        init_centroids=np.empty((n_cluster, n_dim))
        init_centroids_index=list()

        init_centroids_index.append(rnd.randint(n_points))
        init_centroids[0]=X[init_centroids_index[-1]]
       
        for i in range(1,n_cluster):
            
            # new_X=np.delete(X,init_centroids_index,axis=0)
            # no need to create a new array(2D) from X based on only unselected points
            # bcz probability for them would be zero; so they would not be selected again
             
            tmp=np.empty((n_points,i))
            for j in range(i):
                #TODO:done
                # tmp[:,j]=np.sqrt(kernel_dist(X,init_centroids[j,:]))
                tmp[:,j]=np.sqrt(k(X,init_centroids[j,:]))

            tmp_min=np.min(tmp,axis=1) # min(D(X)) for each unselected point

            # convert them into probabilities
            tmp_prob=tmp_min/np.sum(tmp_min)
            # print(np.sum(tmp_min))

            # possible index values
            values=list(range(0,1000))
            # choose an index randomly based on probability
            next_centriod= np.random.choice(a=values, size=1, p=tmp_prob)[0]
            # add next centroid to the list and empty array
            init_centroids_index.append(next_centriod)
            init_centroids[i]=X[init_centroids_index[-1]]
    else:
        print('give a valid choice for init_style')
        return None
    
    '''
    KMeans Iterations
    '''
    curr_centroids=list()
    for i in range(n_cluster):
        curr_centroids.append(init_centroids[i,])

    # to store cluster assignment and error info after each iteration
    cluster_history=list()
    wacc_history=list()

    # for stopping/convergance criterian
    error_tol=10**(-4)
    iter_error_diff=1 
    iter_count=0
    # new_objective_value=sys.float_info.max

    while iter_count<n_iter:
        print('=============')
        # increase iter_count
        iter_count+=1
        
        # create empty temp array to store euclidean dist-sum from cluster centroids
        # do it for each cluster centroid
        tmp_array=np.empty((n_points, n_cluster))

        for cluster in range(n_cluster):
            # here distance is euclidean
            #TODO:done
            tmp_array[:,cluster]=kernel_dist(X,curr_centroids[cluster])
            toc=time()
            print(toc-tic)
        # print(tmp_cluster_assign)
        # assign cluster to each point based on min distance from the cluster
        tmp_cluster_assign=np.argmin(tmp_array,axis=1)
        cluster_history.append(tmp_cluster_assign)
        # print(tmp_cluster_assign)
        # update centeroids by cluster point mean
        curr_objective_value=0
        for cluster in range(n_cluster):
            tmp_cluster_points=X[(tmp_cluster_assign==cluster),:] # 2D array
            #TODO:??
            # tmp_centroid=np.mean(tmp_cluster_points,axis=0) # 1D mean array
            #TODO:done:done
            # print(tmp_cluster_points)
            # curr_objective_value+=np.sum(kernel_dist(tmp_cluster_points,tmp_cluster_points)) # scalar
            curr_centroids[cluster]=tmp_cluster_points # update cluster centroid
        # print(curr_centroids)
        wacc_history.append(curr_objective_value) # add curr error value to history

        # update difference
        # iter_error_diff=new_objective_value-curr_objective_value
        # new_objective_value=curr_objective_value
    return tmp_cluster_assign
    # return cluster ids for each of the data points in X with some history info
    # return {'final':tmp_cluster_assign,'history':cluster_history,'iter-count':iter_count,}#'wacc-history':wacc_history,'final-centroids':curr_centroids,'start-centroid':init_centroids_index}a

tic=time()
plot_clusters(X,get_kernel_KMeans_clusters(X,n_cluster=4,init_style='k-means++',n_iter=100),iter=100,n_cluster=4)
toc=time()

print('total time =',toc-tic)