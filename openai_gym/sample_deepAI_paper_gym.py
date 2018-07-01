"""
	
    Implementation of Deep Active Inference for
    General Artificial Intelligence
    
    Kai Ueltzhoeffer, 2017

"""

# Imports
import cPickle
import timeit
import scipy

from time import sleep

import matplotlib.pyplot as plt

import numpy
import scipy

import theano
import theano.tensor as T
from theano.ifelse import ifelse

from theano import pprint as pp

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams  

import matplotlib.pyplot as plt

import gym
gym.logger.set_level(40)

# Parameters

# For nicer text rendering, requires LateX
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size='16') 

env_name = 'MountainCar-v0'

n_s = 10 # States
base_name = 'deepAI_paper_gym' # Name for Saves and Logfile
learning_rate = 1e-3 # Learning Rate
learning_rate_sigma = 1e-3
saving_steps = 10 # Save progress every nth step
save_best_trajectory = True # Save timecourse of best parameter sets

n_run_steps = 30 # No. of Timesteps to Simulate
n_proc = 1 # No. of Processes to Simulate for each Sample from the Population Density
n_perturbations = 1000 # No. of Samples from Population Density per Iteration

n_o = 2 # Dimensionality of Environment Output (Continous)
n_oa = 3 # Dimensionality of Environment Action Space (Discrete)
n_ha = 10

# Minimum Value of Standard-Deviations, to prevent Division-by-Zero
sig_min_obs = 1e-6
sig_max_obs = 0.05
sig_min_states = 1e-6

init_sig_obs = -3.0
init_sig_states_likelihood = -3.0
init_sig_states = -4.0

sig_min_perturbations = 1e-6
init_sig_perturbations = -3.0

# Max. Number of Optimization Steps
n_steps = 1000000

# Initialize RNG
ii32 = numpy.iinfo(numpy.int32)
theano_rng = RandomStreams(numpy.random.randint(ii32.max)) # ADD RANDOM SEED!  

# Initialize Environments
envs = []
for i in range(n_perturbations):
    envs.append(gym.make(env_name))

# Helper Functions and Classes

# Xavier initialization
def initxavier(shape1, shape2, minval =-0.1, maxval = 0.1):
    
    val = numpy.random.randn(shape1, shape2) / numpy.sqrt(shape2) # "Xavier" initialization
    
    return val.astype(theano.config.floatX) 

# Unitary transform
def initortho(shape1, shape2):
        x = numpy.random.normal(0.0, 0.1, (shape1, shape2))
        xo = scipy.linalg.orth(x)
        
        return xo.astype(theano.config.floatX)

# Uniform Weight Distribution
def initweight(shape1, shape2, minval =-0.05, maxval = 0.05):
    val = numpy.random.rand(
        shape1, shape2
    )
    
    val = minval + (maxval - minval)*val    
    
    return val.astype(theano.config.floatX) 
    
# Constant Weight
def initconst(shape1, shape2, val = 0.0):
    val = val*numpy.ones(
        (shape1,shape2),
        dtype=theano.config.floatX
    )
    
    return val.astype(theano.config.floatX)   

#ADAM Optimizer, following Kingma & Ba (2015), c.f. https://arxiv.org/abs/1412.6980
class Adam(object):

    def __init__(self, grads, p, b1, b2, alpha, epsilon = 10e-8):
        
        # Perform Gradient Clipping
        grad_norm = grads.norm(L=2)
        grads = T.switch( T.lt(1.0, grad_norm), grads/grad_norm, grads)
    
        #self.L = L
        self.p = p
        self.b1 = b1
        self.b2 = b2
        self.alpha = alpha
        
        self.t = theano.shared( value = numpy.cast[theano.config.floatX](1.0))
        self.t_next = self.t + 1
        
        self.g = grads.astype(dtype = theano.config.floatX)
        self.m = theano.shared( value=numpy.zeros_like(
                p.get_value(),
                dtype=theano.config.floatX
            ),
            name='m',
            borrow=True,
            broadcastable=self.p.broadcastable
        )
        self.m_next = self.b1*self.m + (1 - self.b1)*self.g
        self.v = theano.shared( value=numpy.zeros_like(
                p.get_value(),
                dtype=theano.config.floatX
            ),
            name='v',
            borrow=True,
            broadcastable=self.p.broadcastable
        )
        self.v_next = b2*self.v + (1 - self.b2)*self.g*self.g
        self.m_ub = self.m/(1-b1**self.t)
        self.v_ub = self.v/(1-b2**self.t)
        self.update = self.p - alpha*self.m_ub/(T.sqrt(self.v_ub) + epsilon)
        
        self.updates = [(self.t, self.t_next),
                        (self.m, self.m_next),
                        (self.v, self.v_next),
                        (self.p, self.update)]
                             
def GaussianNLL(y, mu, sig):

    nll = 0.5 * T.sum(T.sqr(y - mu) / sig**2 + 2 * T.log(sig) +
                      T.log(2 * numpy.pi), axis=1)
    return nll
    
def KLGaussianGaussian(mu1, sig1, mu2, sig2):
   
    kl = T.sum(0.5 * (2 * T.log(sig2) - 2 * T.log(sig1) +
                   (sig1**2 + (mu1 - mu2)**2) /
                   sig2**2 - 1), axis=1)

    return kl
    
def save_model(params, sigmas, filename):

    with open(filename, 'wb') as f:
        for param in params:
            cPickle.dump(param.get_value(borrow=True), f, -1)
        for sigma in sigmas:
            cPickle.dump(sigma.get_value(borrow=True), f, -1)
            
def load_model(params, sigmas, filename):

    with open(filename, 'r') as f:
        for param in params:
            param.set_value(cPickle.load(f), borrow=True)
        for sigma in sigmas:
            sigma.set_value(cPickle.load(f), borrow=True)
            
def softmax(X):
    eX = T.exp(X - X.max(axis=1, keepdims = True))
    prob = eX / eX.sum(axis=1, keepdims=True)
    return prob      
    
eps = 1e-6
    
def gumbel(shape):
    U = theano_rng.uniform(shape, low=eps, high=1.0-eps, dtype=theano.config.floatX )    
    return  -T.log(-T.log(U))
 
def Cat_sample(pi, num_sample=None):
    pi2 = T.clip(pi, eps, 1.0 - eps)
    y = T.log(pi2) + gumbel(pi.shape)    
    
    return T.cast( T.eq(y, y.max(axis = 1, keepdims = True)), dtype = theano.config.floatX )    

#########################################
#
# Parameters
#
#########################################

# Generate List of Parameters to Optimize
params = []

# Parameters of approximate posterior

# Q(s_t | s_t-1, o_t, oh_t, oa_t)

Wq_hst_ot = theano.shared(
    value=initweight(n_s, n_o, -10.0, 10.0).reshape(1,n_s,n_o),
    name='Wq_hst_ot',
    borrow=True,
    broadcastable=(True, False, False)
    
)

params.append(Wq_hst_ot)

Wq_hst_stm1 = theano.shared(
    value=initweight(n_s, n_s, -0.01, 0.01).reshape(1,n_s,n_s),#initortho(n_s, n_s).reshape(1,n_s,n_s),
    name='Wq_hst_stm1',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wq_hst_stm1)

bq_hst = theano.shared(
    value=initconst(n_s, 1).reshape(1,n_s,1),
    name='bq_hst',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(bq_hst)

Wq_hst2_hst = theano.shared(
    value=initweight(n_s, n_s).reshape(1, n_s, n_s),
    name='Wq_hst2_hst',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wq_hst2_hst)

bq_hst2 = theano.shared(
    value=initconst(n_s, 1).reshape(1,n_s,1),
    name='bq_hst2',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(bq_hst2)

Wq_stmu_hst2 = theano.shared(
    value=initweight(n_s, n_s).reshape(1,n_s,n_s),
    name='Wq_stmu_hst2',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wq_stmu_hst2)

bq_stmu = theano.shared(
    value=initconst(n_s, 1).reshape(1,n_s,1),
    name='bq_stmu',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(bq_stmu)

Wq_stsig_hst2 = theano.shared(
    value=initweight(n_s, n_s).reshape(1,n_s,n_s),
    name='Wq_stsig_hst2',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wq_stsig_hst2)

bq_stsig = theano.shared(
    value=initconst(n_s,1,init_sig_states).reshape(1,n_s,1),
    name='bq_stsig',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(bq_stsig)

# Define Parameters for Likelihood Function

# p( s_t | s_t-1 )

Wl_stmu_stm1 = theano.shared(
    value=initweight(n_s, n_s).reshape(1,n_s,n_s),#initortho(n_s, n_s).reshape(1,n_s,n_s),
    name='Wl_stmu_stm1',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wl_stmu_stm1)

bl_stmu = theano.shared(
    value=initconst(n_s, 1).reshape(1,n_s,1),
    name='bl_stmu',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(bl_stmu)

Wl_stsig_stm1 = theano.shared(
    value=initweight(n_s, n_s).reshape(1,n_s,n_s),
    name='Wl_stsig_stm1',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wl_stsig_stm1)

bl_stsig = theano.shared(
    value=initconst(n_s, 1,init_sig_states_likelihood).reshape(1,n_s,1),
    name='bl_stsig',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(bl_stsig)

# p( o_t | s_t )

Wl_ost_st = theano.shared(
    value=initweight(n_s, n_s).reshape(1,n_s,n_s),
    name='Wl_ost_st',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wl_ost_st)

bl_ost = theano.shared(
    value=initconst(n_s, 1).reshape(1,n_s,1),
    name='bl_ost',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(bl_ost)

Wl_otmu_st = theano.shared(
    value=initweight(n_o, n_s).reshape(1,n_o,n_s),
    name='Wl_otmu_st',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wl_otmu_st)

bl_otmu = theano.shared(
    value=initconst(n_o, 1).reshape(1,n_o,1),
    name='bl_otmu',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(bl_otmu)

Wl_otsig_st = theano.shared(
    value=initweight(n_o, n_s).reshape(1,n_o,n_s),
    name='Wl_otsig_st',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wl_otsig_st)

bl_otsig = theano.shared(
    value=initconst(n_o,1,init_sig_obs).reshape(1,n_o,1),
    name='bl_otsig',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(bl_otsig)

# Action Function

Wa_aht_st = theano.shared(
    value=initweight(n_ha, n_s, -10.0, 10.).reshape(1,n_ha,n_s),
    name='Wa_aht_st',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wa_aht_st)

ba_aht = theano.shared(
    value=initconst(n_ha, 1).reshape(1,n_ha,1),
    name='ba_aht',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(ba_aht)

Wa_aht2_aht = theano.shared(
    value=initweight(n_ha, n_ha).reshape(1,n_ha,n_ha),
    name='Wa_aht2_aht',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wa_aht2_aht)

ba_aht2 = theano.shared(
    value=initconst(n_ha, 1).reshape(1,n_ha,1),
    name='ba_aht2',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(ba_aht2)

Wa_aht3_aht2 = theano.shared(
    value=initweight(n_ha, n_ha).reshape(1,n_ha,n_ha),
    name='Wa_aht3_aht2',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wa_aht3_aht2)

ba_aht3 = theano.shared(
    value=initconst(n_ha, 1).reshape(1,n_ha,1),
    name='ba_aht3',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(ba_aht3)

Wa_atpi_aht = theano.shared(
    value=initweight(n_oa, n_ha, -100.0, 100.0).reshape(1,n_oa,n_ha),
    name='Wa_atpi_aht',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wa_atpi_aht)

ba_atpi = theano.shared(
    value=initconst(n_oa, 1).reshape(1,n_oa,1),
    name='ba_atpi',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(ba_atpi)

####################################################################
#
# Function to Create randomly perturbed version of parameters
#
####################################################################

# Initialize Sigmas

sigmas = []

for param in params:
    sigma = theano.shared(name = 'sigma_' + param.name, value = init_sig_perturbations*numpy.ones( param.get_value().shape ).astype( dtype = theano.config.floatX ), borrow = True, broadcastable=param.broadcastable )
    sigmas.append(sigma)

'''
for i in range(len(params)):
    if i >= len(params) - 4:
        sigma = theano.shared(name = 'sigma_' + params[i].name, value = 0.1*init_sig_perturbations*numpy.ones( params[i].get_value().shape ).astype( dtype = theano.config.floatX ), borrow = True, broadcastable=params[i].broadcastable )
    else:
        sigma = theano.shared(name = 'sigma_' + params[i].name, value = init_sig_perturbations*numpy.ones( params[i].get_value().shape ).astype( dtype = theano.config.floatX ), borrow = True, broadcastable=params[i].broadcastable )
    
    sigmas.append(sigma)
'''                
# Create placeholders for current perturbations of variables

load_model(params, sigmas, base_name + '_best.pkl')

base_name = base_name + '_sample'
       
r_params = []
r_epsilons = []

for param in params:
	
	r_params.append( theano.shared(name = 'r_' + param.name, value = numpy.zeros( (n_perturbations, param.get_value().shape[1], param.get_value().shape[2] ) ).astype( dtype = theano.config.floatX ), borrow = True, broadcastable=(False, param.broadcastable[1], param.broadcastable[2]) ) )
	r_epsilons.append( theano.shared(name = 'r_epsilon_' + param.name, value = numpy.zeros( (n_perturbations, param.get_value().shape[1], param.get_value().shape[2] ) ).astype( dtype = theano.config.floatX ), borrow = True, broadcastable=(False, param.broadcastable[1], param.broadcastable[2]) ) )
	
# Create update to randomize shared variable representation of samples from population density on parameters
	
updates_randomize_params = []

for i in range(len(params)):
	
    epsilon_half = theano_rng.normal((n_perturbations/2,params[i].shape[1],params[i].shape[2]), dtype = theano.config.floatX)
    r_epsilon = T.concatenate( [epsilon_half, -1.0*epsilon_half], axis = 0 )
    r_param = params[i] + r_epsilon*(T.nnet.softplus( sigmas[i] ) + sig_min_perturbations)
    updates_randomize_params.append( (r_params[i], r_param) )
    updates_randomize_params.append( (r_epsilons[i], r_epsilon) )
	
randomize_params = theano.function(inputs = [], outputs = [], updates = updates_randomize_params)

print 'randomize_params compiled...'

updates_maximize_params = []

for i in range(len(params)):
    updates_maximize_params.append( (r_params[i], 0.0*r_params[i] + params[i]) )

maximize_params = theano.function(inputs = [], outputs = [], updates = updates_maximize_params)

print 'maximize_params compiled...'

'''
t_start = timeit.default_timer()

for i in range(100):
    randomize_params()
    
t_end = timeit.default_timer()

print '100 randomizations took %f seconds.' % (t_end - t_start)
'''

####################################################################
#
# Name the randomly perturbed version of parameters
#
####################################################################
    
[r_Wq_hst_ot, r_Wq_hst_stm1, r_bq_hst,\
r_Wq_hst2_hst, r_bq_hst2,\
r_Wq_stmu_hst2, r_bq_stmu,\
r_Wq_stsig_hst2, r_bq_stsig,\
r_Wl_stmu_stm1, r_bl_stmu,\
r_Wl_stsig_stm1, r_bl_stsig,\
r_Wl_ost_st, r_bl_ost,\
r_Wl_otmu_st, r_bl_otmu,\
r_Wl_otsig_st, r_bl_otsig,\
r_Wa_aht_st, r_ba_aht,\
r_Wa_aht2_aht, r_ba_aht2,\
r_Wa_aht3_aht2, r_ba_aht3,\
r_Wa_atpi_aht, r_ba_atpi] = r_params

###################################################################
#  
# Define Variational Free Energy for Simulated Run
#
###################################################################

# NOTE: 
# As we want to interface with OpenAI gym, we have to be able to
# input new observations and output actions for every single
# timestep of each process. Thus we formulate the sampling of the 
# variational free energy in terms of individual update steps performed
# on shared variables.

###################################################################
#
# Initialize shared variables
#
###################################################################

st = theano.shared(name = 's_t0', value = numpy.zeros( (n_perturbations,n_s, n_proc) ).astype( dtype = theano.config.floatX ), borrow = True, broadcastable=(False, False, True))

FA = theano.shared(name = 's_t0', value = numpy.zeros( (n_perturbations,n_proc) ).astype( dtype = theano.config.floatX ), borrow = True, broadcastable=(False, False))

mask_FA = theano.shared(name = 's_t0', value = numpy.ones( (n_perturbations,n_proc) ).astype( dtype = theano.config.floatX ), borrow = True, broadcastable=(False, False))

pA = theano.shared(name = 's_t0', value = numpy.zeros( (n_perturbations,n_proc) ).astype( dtype = theano.config.floatX ), borrow = True, broadcastable=(False, False))
KLA = theano.shared(name = 's_t0', value = numpy.zeros( (n_perturbations,n_proc) ).astype( dtype = theano.config.floatX ), borrow = True, broadcastable=(False, False))

ot = T.ftensor3()
rewt = T.ftensor3()
sig_rew_th = T.fscalar()

hst =  T.tanh( T.batched_tensordot(r_Wq_hst_stm1,T.reshape(st,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + T.batched_tensordot(r_Wq_hst_ot,T.reshape(ot,(n_perturbations,n_o,n_proc)),axes=[[2],[1]]) + r_bq_hst )
hst2 =  T.tanh( T.batched_tensordot(r_Wq_hst2_hst,T.reshape(hst,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_bq_hst2 )

stmu =  T.tanh( T.batched_tensordot(r_Wq_stmu_hst2,T.reshape(hst2,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_bq_stmu )
stsig = T.nnet.sigmoid( T.batched_tensordot(r_Wq_stsig_hst2,T.reshape(hst2,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_bq_stsig ) + sig_min_states

# Explicitly encode reward as homeostatic state variable
# Rescale representation to fit within linear response of the tanh-nonlinearity
# stmu = T.set_subtensor(stmu[:,0,:],0.1*rewt[:,0,:]).reshape((n_perturbations,n_s,n_proc))
# stsig = T.set_subtensor(stsig[:,0,:],0.01).reshape((n_perturbations,n_s,n_proc))

# Sample from variational density
stp1 = stmu + theano_rng.normal((n_perturbations,n_s,n_proc))*stsig
    
# Calculate parameters of likelihood distributions from sampled state
ost =  T.tanh( T.batched_tensordot(r_Wl_ost_st, T.reshape(stp1,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_bl_ost )
otmu = T.batched_tensordot(r_Wl_otmu_st, T.reshape(ost,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_bl_otmu
otsig = (sig_max_obs - sig_min_obs)*T.nnet.sigmoid(T.batched_tensordot(r_Wl_otsig_st, T.reshape(ost,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_bl_otsig) + sig_min_obs
     
# Calculate negative log-likelihood of observations
p_ot  = GaussianNLL(ot, otmu, otsig)

# Explicit expectations on reward channel: should always be +1 :o)
p_rew = 0.0*GaussianNLL(rewt, 1.0, 1.0)

# Use hidden state to generate action state
aht = T.batched_tensordot(r_Wa_aht_st, T.reshape(stp1,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_ba_aht
aht2 = T.tanh( T.batched_tensordot(r_Wa_aht2_aht, T.reshape(aht,(n_perturbations,n_ha,n_proc)),axes=[[2],[1]]) + r_ba_aht2 )
aht3 = T.tanh( T.batched_tensordot(r_Wa_aht3_aht2, T.reshape(aht2,(n_perturbations,n_ha,n_proc)),axes=[[2],[1]]) + r_ba_aht3 )
atpi = softmax( T.batched_tensordot(r_Wa_atpi_aht, T.reshape(aht3,(n_perturbations,n_ha,n_proc)),axes=[[2],[1]]) + r_ba_atpi )

at = atpi # Cat_sample(atpi) #DETERMINISTIC ACTION!

# Calculate prior expectation on hidden state from previous state
prior_stmu = T.tanh( T.batched_tensordot(r_Wl_stmu_stm1, T.reshape(st,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_bl_stmu )
prior_stsig = T.nnet.sigmoid( T.batched_tensordot(r_Wl_stsig_stm1, T.reshape(st,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_bl_stsig ) + sig_min_states
    
# Explicitly encode expectations on homeostatic state variable
# prior_stmu = T.set_subtensor(prior_stmu[:,0,:],0.1)         #ifelse(T.lt(t,20),prior_stmu, T.set_subtensor(prior_stmu[:,0,:],0.1))
# prior_stsig = T.set_subtensor(prior_stsig[:,0,:],0.01)     #ifelse(T.lt(t,20),prior_stsig, T.set_subtensor(prior_stsig[:,0,:],0.005))    
   
# Calculate KL divergence between variational density and prior density
# using explicit formula for diagonal gaussians
KL_st = KLGaussianGaussian(stmu, stsig, prior_stmu, prior_stsig)
    
# Put free energy functional together
FEt =  p_rew + KL_st + p_ot

update_FA = [ (FA, FA + mask_FA*FEt) ]
update_st = [ (st, stp1 )]
update_diagnostics = [ (pA,pA + p_ot), (KLA, KLA + KL_st)]

step_agent = theano.function(inputs = [ot, rewt], outputs = [at, p_rew, FEt, otmu, otsig, ot, stmu, stsig, stp1, prior_stmu, prior_stsig], updates = update_st + update_FA + update_diagnostics, on_unused_input='ignore', allow_input_downcast = True)

reset_FA = theano.function(inputs = [], outputs = [], updates = [(FA, 0.0*FA)])
reset_st = theano.function(inputs = [], outputs = [], updates = [(st, 0.0*st)])
reset_diagnostics = theano.function(inputs = [], outputs = [], updates = [(pA,0.0*pA), (KLA, 0.0*KLA)])

obs = numpy.zeros( (n_perturbations, n_o, n_proc ), dtype = theano.config.floatX )
rews = numpy.zeros( (n_perturbations, 1, n_proc ), dtype = theano.config.floatX )

mean_rews = numpy.zeros( (n_perturbations, ) )
dones = numpy.zeros( (n_perturbations, ), dtype = bool )

# Additional sampling quantities! :o)

prior_stp1 = prior_stmu + theano_rng.normal((n_perturbations,n_s,n_proc))*prior_stsig

prior_ost =  T.tanh( T.batched_tensordot(r_Wl_ost_st, T.reshape(st,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_bl_ost )
prior_otmu = T.batched_tensordot(r_Wl_otmu_st, T.reshape(prior_ost,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_bl_otmu
prior_otsig = (sig_max_obs - sig_min_obs)*T.nnet.sigmoid(T.batched_tensordot(r_Wl_otsig_st, T.reshape(prior_ost,(n_perturbations,n_s,n_proc)),axes=[[2],[1]]) + r_bl_otsig) + sig_min_obs

s_ot = prior_otmu + theano_rng.normal((n_perturbations,n_o,n_proc))*prior_otsig

update_sample = [ (st, prior_stp1) ]

sample_timestep = theano.function(inputs = [], outputs = [s_ot, prior_otmu, prior_otsig], updates = update_sample, on_unused_input='ignore', allow_input_downcast = True)

def do_sampling(n_steps):
    
    reset_st()
    
    observations = numpy.zeros( (n_steps, n_perturbations, n_o, n_proc) )
    full_prior_otmu = numpy.zeros( (n_steps, n_perturbations, n_o, n_proc) )
    full_prior_otsig = numpy.zeros( (n_steps, n_perturbations, n_o, n_proc) )
    
    for i in range(n_steps):
        [obs, prior_otmu, prior_otsig] = sample_timestep()
        observations[i] = obs
        full_prior_otmu[i] = prior_otmu
        full_prior_otsig[i] = prior_otsig
        
    return observations, full_prior_otmu, full_prior_otsig
        
    
    

# Initialize environments
    
def evaluate_FA():
    
    reset_st()
    reset_FA()
    reset_diagnostics()
    mask_FA.set_value( numpy.ones( mask_FA.get_value().shape, dtype = theano.config.floatX ) )
    
    # Reset environments and collect first obsevations
    for i in range(n_perturbations):
        ob = envs[i].reset()
        obs[i,:,0] = ob
        rews[i,0,0] = -1.0        
        dones[i] = False
        mean_rews[i] = 0.0
        
    #print [FA.get_value().mean(), mean_rews.mean(), KLA.get_value().mean(), pA.get_value().mean(), KLhA.get_value().mean()]

        
    # Sample first actions
    [act, op_rew, oFET, otmu, otsig, oot, stmu, stsig, ost, prior_stmu, prior_stsig] = step_agent(obs, rews)
    
    # As long as not all processes are finished
    while numpy.any( dones == False ):
        
        # Iterate over processes
        for i in range(n_perturbations):
            
            # If process is not finished
            if dones[i] == False:
                
                # Do environmental step with sampled action,
                # collectiong new observations
                ob, rew, done, _  = envs[i].step( numpy.argmax( act[i,:,0].squeeze() ) )
                obs[i,:,0] = ob
                rews[i,0,0] = rew
                dones[i] = done
            
                mean_rews[i] = mean_rews[i] + rew
                
                if done:
                    temp = mask_FA.get_value()
                    temp[i,:] = 0.0
                    mask_FA.set_value(temp)
        
        # Sample new actions  
        [act, op_rew, oFEt, otmu, otsig, oot, stmu, stsig, ost, prior_stmu, prior_stsig] = step_agent(obs, rews)
    
    oFA = FA.get_value().mean()
    omean_rews = mean_rews.mean()
    oKLA = KLA.get_value().mean()
    opA = pA.get_value().mean()
    
    if omean_rews > 100.0:
        print '###################################################################'            
    print [oFA, omean_rews, oKLA, opA]
    #print op_rew.shape
    #print op_rew.mean()
    #print oFEt.shape
    #print oFEt.mean()
    if mean_rews.mean() > 100.0:
        print '###################################################################'       
        
    return oFA, omean_rews, oKLA, opA
    
def simulate():
    
    reset_st()
    reset_FA()
    reset_diagnostics()
    mask_FA.set_value( numpy.ones( mask_FA.get_value().shape, dtype = theano.config.floatX ) )
    
    full_otmu = numpy.zeros( (200, n_perturbations, n_o, n_proc) )
    full_otsig = numpy.zeros( (200, n_perturbations, n_o, n_proc) )
    full_ot = numpy.zeros( (200, n_perturbations, n_o, n_proc) )
    
    full_stmu = numpy.zeros( (200, n_perturbations, n_s, n_proc) )
    full_stsig = numpy.zeros( (200, n_perturbations, n_s, n_proc) )
    full_st = numpy.zeros( (200, n_perturbations, n_s, n_proc) )
    
    full_prior_stmu = numpy.zeros( (200, n_perturbations, n_s, n_proc) )
    full_prior_stsig = numpy.zeros( (200, n_perturbations, n_s, n_proc) )
    
    # Reset environments and collect first obsevations
    for i in range(n_perturbations):
        ob = envs[i].reset()
        obs[i,:,0] = ob
        rews[i,0,0] = -1.0        
        dones[i] = False
        mean_rews[i] = 0.0
        
    #print [FA.get_value().mean(), mean_rews.mean(), KLA.get_value().mean(), pA.get_value().mean(), KLhA.get_value().mean()]

        
    # Sample first actions
    [act, op_rew, oFET, otmu, otsig, oot, stmu, stsig, ost, prior_stmu, prior_stsig] = step_agent(obs, rews)
    
    timestep = 0
    
    # As long as not all processes are finished
    while numpy.any( dones == False ):
        
        # Iterate over processes
        for i in range(n_perturbations):
            
            # If process is not finished
            if dones[i] == False:
                
                #if i == 0:
                #    envs[i].render()
                #    sleep(0.05)
                
                # Do environmental step with sampled action,
                # collectiong new observations
                ob, rew, done, _  = envs[i].step( numpy.argmax( act[i,:,0].squeeze() ) )
                obs[i,:,0] = ob
                rews[i,0,0] = rew
                dones[i] = done
            
                mean_rews[i] = mean_rews[i] + rew
                
                if done:
                    temp = mask_FA.get_value()
                    temp[i,:] = 0.0
                    mask_FA.set_value(temp)
        
        # Sample new actions  
        [act, op_rew, oFEt, otmu, otsig, oot, stmu, stsig, ost, prior_stmu, prior_stsig] = step_agent(obs, rews)
        
        full_otmu[timestep] = otmu
        full_otsig[timestep] = otsig
        full_ot[timestep] = oot
        
        full_stmu[timestep] = stmu
        full_stsig[timestep] = stsig
        full_st[timestep] = ost
        
        full_prior_stmu[timestep] = prior_stmu
        full_prior_stsig[timestep] = prior_stsig
    
        timestep = timestep + 1
        print 'timestep: %d' % timestep
    
    oFA = FA.get_value().mean()
    omean_rews = mean_rews.mean()
    oKLA = KLA.get_value().mean()
    opA = pA.get_value().mean()
    
    if omean_rews > 100.0:
        print '###################################################################'            
    print [oFA, omean_rews, oKLA, opA]
    #print op_rew.shape
    #print op_rew.mean()
    #print oFEt.shape
    #print oFEt.mean()
    if mean_rews.mean() > 100.0:
        print '###################################################################'       
        
    return full_otmu, full_otsig, full_ot, full_stmu, full_stsig, full_st, full_prior_stmu, full_prior_stsig

'''        
t_start = timeit.default_timer()

evaluate_FA()

t_end = timeit.default_timer()

print 'Took %f seconds!' % (t_end - t_start)  
'''

    
########################################################################
#
#
# Put everything together
#
#
########################################################################


print 'Population:'
randomize_params()
evaluate_FA()
print 'Mean:'
maximize_params()
evaluate_FA()

print 'Running'
full_otmu, full_otsig, full_ot, full_stmu, full_stsig, full_st, full_prior_stmu, full_prior_stsig = simulate()

plt.plot(full_otmu[:,0,0,0],label  ='otmu')
plt.plot(full_otsig[:,0,0,0],label  ='otsig')
plt.plot(full_ot[:,0,0,0],label  ='ot')
plt.plot(full_otmu[:,0,1,0],label  ='otmu2')
plt.plot(full_otsig[:,0,1,0],label  ='otsig2')
plt.plot(full_ot[:,0,1,0],label  ='ot2')
plt.legend()
plt.show()

plt.subplot(231)
plt.plot(full_stmu[:,0,:,0],label  ='stmu')
plt.legend()
plt.subplot(232)
plt.plot(full_stsig[:,0,:,0],label  ='stmu')
plt.subplot(233)
plt.plot(full_st[:,0,:,0],label  ='st')
plt.subplot(234)
plt.plot(full_prior_stmu[:,0,:,0],label  ='prior_stmu')
plt.subplot(235)
plt.plot(full_prior_stsig[:,0,:,0],label  ='prior_stsig')
#plt.plot(full_stsig[:,0,0,0],label  ='stsig')
#plt.plot(full_st[:,0,0,0],label  ='st')
#plt.plot(full_stmu[:,0,1,0],label  ='stmu2')
#plt.plot(full_stsig[:,0,1,0],label  ='stsig2')
#plt.plot(full_st[:,0,1,0],label  ='st2')
plt.legend()
plt.show()

print 'Sampling:'
obs, ootmu, ootsig = do_sampling(200)
    
#plt.subplot(224)
plt.plot(obs[:,:10,0,0],label = 'cart pos') 
plt.plot(obs[:,0,1,0],label = 'cart vel')
plt.plot(ootmu[:,0,0,0],label = 'cart pos mu') 
plt.plot(ootsig[:,0,0,0],label = 'cart pos sig')
plt.plot(ootmu[:,0,1,0],label = 'cart vel mu') 
plt.plot(ootsig[:,0,1,0],label = 'cart vel sig')
plt.grid()
plt.xlabel('Steps')
plt.legend()

plt.show()

fig = plt.figure(3,figsize=(12,12))
plt.subplot(2,2,1)
plt.title('$o_x(t)$')
for j in range(10):
    plt.plot(full_ot[:,j,0,0].squeeze())
plt.xlabel('t / Steps')
plt.ylabel('$o_x$ / a.u.')
plt.subplot(2,2,2)
plt.title('$o_v(t)$')
for j in range(10):
    plt.plot(full_ot[:,j,1,0].squeeze())
plt.xlabel('t / Steps')
plt.ylabel('$o_v$ / a.u.')
plt.subplot(2,2,3)
plt.title('$o_x(t)$ sampled')
for j in range(3):
    plt.plot(obs[:,j,0,0].squeeze())
plt.xlabel('t / Steps')
plt.ylabel('$o_x$ / a.u.')
plt.subplot(2,2,4)
plt.title('$o_v(t)$ sampled')
for j in range(3):
    plt.plot(obs[:,j,1,0].squeeze())
plt.xlabel('t / Steps')
plt.ylabel('$o_v$ / a.u.')

fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95,
                    wspace=0.3, hspace=0.3)
fig.savefig('fig_openai_mountaincar.pdf')

plt.show()

