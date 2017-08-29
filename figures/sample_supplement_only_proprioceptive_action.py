"""
	
    Implementation of Deep Active Inference for
    General Artificial Intelligence
    
    Kai Ueltzhoeffer, 2017

"""

# Imports
import cPickle
import timeit
import scipy

import matplotlib.pyplot as plt

import numpy
import scipy

import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano import pprint as pp

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams 

# Filename of stored model parameters
filename = 'supplement_only_proprioceptive_action_best.pkl'

# For nicer text rendering, requires LateX
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size='16') 

# Sampling Parameters 

# Constrained Sampling(_ag means "actions given")
n_iterations_ag = 100  # How many iterations of MCMC to perform for 
                       # constrained sampling (100 is considerably faster)
shift_ag = 10 # Offset to shift action timecourse for constrained sampling

# Parameters (Should be identical to fitting script)
n_s = 10 # States

n_o = 1 # Sensory Input encoding Position
n_oh = 1 # Nonlinearly Transformed Channel (OPTIONAL!)
n_oa = 1 # Proprioception (OPTIONAL!)

n_run_steps = 30
n_proc = 10

n_steps = 1000000

n_sample_steps = 30
n_samples = 10

# Minimum Value of Standard-Deviations, to prevent Division-by-Zero
sig_min_obs = 1e-6
sig_min_states = 1e-6
sig_min_action = 1e-6

init_sig_obs = 0.0
init_sig_states_likelihood = 0.0
init_sig_states = -3.0
init_sig_action = -3.0

sig_min_perturbations = 1e-6
init_sig_perturbations = -3.0

# Initialize RNG
ii32 = numpy.iinfo(numpy.int32)
theano_rng = RandomStreams(numpy.random.randint(ii32.max)) # ADD RANDOM SEED!  

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

def GaussianNLL(y, mu, sig):

    nll = 0.5 * T.sum(T.sqr(y - mu) / sig**2 + 2 * T.log(sig) +
                      T.log(2 * numpy.pi), axis=0)
    return nll   

def KLGaussianGaussian(mu1, sig1, mu2, sig2):
   
    kl = T.sum(0.5 * (2 * T.log(sig2) - 2 * T.log(sig1) +
                   (sig1**2 + (mu1 - mu2)**2) /
                   sig2**2 - 1), axis=0)

    return kl
    
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
    value=initweight(n_s, n_o, -0.5, 0.5).reshape(1,n_s,n_o),
    name='Wq_hst_ot',
    borrow=True,
    broadcastable=(True, False, False)
    
)

params.append(Wq_hst_ot)

Wq_hst_oht = theano.shared(
    value=initweight(n_s, n_oh, -0.5, 0.5).reshape(1,n_s,n_oh),
    name='Wq_hst_oht',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wq_hst_oht)

Wq_hst_oat = theano.shared(
    value=initweight(n_s, n_oa, -0.5, 0.5).reshape(1,n_s,n_oa),
    name='Wq_hst_oat',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wq_hst_oat)

Wq_hst_stm1 = theano.shared(
    value=initortho(n_s, n_s).reshape(1,n_s,n_s),
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
    value=initortho(n_s, n_s).reshape(1, n_s, n_s),
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
    value=initortho(n_s, n_s).reshape(1,n_s,n_s),
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
    value=initweight(n_s, n_s).reshape(1,n_s,n_s),#Wq_stmu_stm1.get_value(),#initortho(n_s, n_s),
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
    value=initweight(n_s, n_s).reshape(1,n_s,n_s),#Wq_stsig_stm1.get_value(),#initweight(n_s, n_s),
    name='Wl_stsig_stm1',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wl_stsig_stm1)

bl_stsig = theano.shared(
    value=initconst(n_s, 1,init_sig_states).reshape(1,n_s,1),
    name='bl_stsig',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(bl_stsig)

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

Wl_ost2_ost = theano.shared(
    value=initweight(n_s, n_s).reshape(1,n_s,n_s),
    name='Wl_ost2_ost',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wl_ost2_ost)

bl_ost2 = theano.shared(
    value=initconst(n_s, 1).reshape(1,n_s,1),
    name='bl_ost2',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(bl_ost2)

Wl_ost3_ost2 = theano.shared(
    value=initweight(n_s, n_s).reshape(1,n_s,n_s),
    name='Wl_ost3_ost2',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wl_ost3_ost2)

bl_ost3 = theano.shared(
    value=initconst(n_s, 1).reshape(1,n_s,1),
    name='bl_ost3',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(bl_ost3)

# p( o_t | s_t )

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

# p( oh_t | s_t )

Wl_ohtmu_st = theano.shared(
    value=initweight(n_oh, n_s).reshape(1,n_oh,n_s),
    name='Wl_ohtmu_st',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wl_ohtmu_st)

bl_ohtmu = theano.shared(
    value=initconst(n_oh, 1).reshape(1,n_oh,1),
    name='bl_ohtmu',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(bl_ohtmu)

Wl_ohtsig_st = theano.shared(
    value=initweight(n_oh, n_s).reshape(1,n_oh,n_s),
    name='Wl_ohtsig_st',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wl_ohtsig_st)

bl_ohtsig = theano.shared(
    value=initconst(n_oh, 1,init_sig_obs).reshape(1,n_oh,1),
    name='bl_ohtsig',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(bl_ohtsig)

# p( oa_t | s_t, a_t )

Wl_oatmu_st = theano.shared(
    value=initweight(n_oa, n_s).reshape(1,n_oa,n_s),
    name='Wl_oatmu_st',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wl_oatmu_st)

bl_oatmu = theano.shared(
    value=initconst(n_oa, 1).reshape(1,n_oa,1),
    name='bl_oatmu',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(bl_oatmu)

Wl_oatsig_st = theano.shared(
    value=initweight(n_oa, n_s).reshape(1,n_oa,n_s),
    name='Wl_oatsig_st',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wl_oatsig_st)

bl_oatsig = theano.shared(
    value=initconst(n_oa, 1,init_sig_obs).reshape(1,n_oa,1),
    name='bl_oatsig',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(bl_oatsig)
# Action Function

Wa_aht_st = theano.shared(
    value=initortho(n_s, n_s).reshape(1,n_s,n_s),
    name='Wa_aht_st',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wa_aht_st)

ba_aht = theano.shared(
    value=initconst(n_s, 1).reshape(1,n_s,1),
    name='ba_aht',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(ba_aht)
'''
Wa_aht2_aht = theano.shared(
    value=initortho(n_s, n_s).reshape(1,n_s,n_s),
    name='Wa_aht2_aht',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wa_aht2_aht)

ba_aht2 = theano.shared(
    value=initconst(n_s, 1).reshape(1,n_s,1),
    name='ba_aht2',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(ba_aht2)

Wa_aht3_aht2 = theano.shared(
    value=initortho(n_s, n_s).reshape(1,n_s,n_s),
    name='Wa_aht3_aht2',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wa_aht3_aht2)

ba_aht3 = theano.shared(
    value=initconst(n_s, 1).reshape(1,n_s,1),
    name='ba_aht3',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(ba_aht3)
'''
Wa_atmu_aht = theano.shared(
    value=initweight(n_oa, n_s,-1.0,1.0).reshape(1,n_oa,n_s),
    name='Wa_atmu_aht',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wa_atmu_aht)

ba_atmu = theano.shared(
    value=initconst(n_oa, 1).reshape(1,n_oa,1),
    name='ba_atmu',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(ba_atmu)

Wa_atsig_aht = theano.shared(
    value=initweight(n_oa, n_s).reshape(1,n_oa,n_s),
    name='Wa_atsig_aht',
    borrow=True,
    broadcastable=(True, False, False)
)

params.append(Wa_atsig_aht)

ba_atsig = theano.shared(
    value=initconst(n_oa, 1,init_sig_action).reshape(1,n_oa,1),
    name='ba_atsig',
    borrow=True,
    broadcastable=(True,False,True)
)

params.append(ba_atsig)

####################################################################
#
# Initialize Standard-Deviations
#
####################################################################

def initialize_sigmas(params, init_sig_perturbations):
    
    sigmas = []
    
    for param in params:
        sigma = theano.shared(name = 'sigma_' + param.name, value = init_sig_perturbations*numpy.ones( param.get_value().shape ).astype( dtype = theano.config.floatX ), borrow = True, broadcastable=param.broadcastable )
        sigmas.append(sigma)
        
    return sigmas
    
sigmas = initialize_sigmas(params, init_sig_perturbations)
       
#########################################
#
# Load Parameters from File
#
#########################################

def load_model(params, sigmas, filename):

    with open(filename, 'r') as f:
        for param in params:
            param.set_value(cPickle.load(f), borrow=True)
        for sigma in sigmas:
            sigma.set_value(cPickle.load(f), borrow=True)

def reduce_first_dim(x):
    
    ms = []
    for m in x:
        m = m.reshape( (m.shape[1], m.shape[2]) )
        ms.append(m)
    return ms

# Load learned Agent
load_model(params, sigmas, filename)

[Wq_hst_ot, Wq_hst_oht, Wq_hst_oat, Wq_hst_stm1, bq_hst,\
Wq_hst2_hst, bq_hst2,\
Wq_stmu_hst2, bq_stmu,\
Wq_stsig_hst2, bq_stsig,\
Wl_stmu_stm1, bl_stmu,\
Wl_stsig_stm1, bl_stsig,\
Wl_ost_st, bl_ost,\
Wl_ost2_ost, bl_ost2,\
Wl_ost3_ost2, bl_ost3,\
Wl_otmu_st, bl_otmu,\
Wl_otsig_st, bl_otsig,\
Wl_ohtmu_st, bl_ohtmu,\
Wl_ohtsig_st, bl_ohtsig,\
Wl_oatmu_st, bl_oatmu,\
Wl_oatsig_st, bl_oatsig,\
Wa_aht_st, ba_aht,\
Wa_atmu_aht, ba_atmu,\
Wa_atsig_aht, ba_atsig\
] = reduce_first_dim(params)

###################################################################
#  
# Define Variational Free Energy for Simulated Run
#
###################################################################

# Build sampling function
def inner_fn(t, stm1, postm1, vtm1):
   
    # Use hidden state to generate action state
    aht = T.dot(Wa_aht_st, T.reshape(stm1,(n_s,n_proc))) + ba_aht
    #aht2 = T.dot(Wa_aht2_aht, T.reshape(aht,(n_s,n_proc))) + ba_aht2
    #aht3 = T.dot(Wa_aht3_aht2, T.reshape(aht2,(n_s,n_proc))) + ba_aht3
    atm1_mu = T.dot(Wa_atmu_aht, T.reshape(aht,(n_s,n_proc))) + ba_atmu
    atm1_sig = T.nnet.softplus( T.dot(Wa_atsig_aht, T.reshape(aht,(n_s,n_proc))) + ba_atsig ) + sig_min_action
    
    # Sample Action
    atm1 = atm1_mu + theano_rng.normal((n_oa,n_proc))*atm1_sig
    
    # Update Environment
    action_force = T.tanh( atm1 )
    force = T.switch(T.lt(postm1,0.0),-2*postm1 - 1,-T.pow(1+5*T.sqr(postm1),-0.5)-T.sqr(postm1)*T.pow(1 + 5*T.sqr(postm1),-1.5)-T.pow(postm1,4)/16.0) - 0.25*vtm1
    vt = vtm1 + 0.05*force + 0.03*action_force
    post = postm1 + vt     
    
    # Generate Sensory Inputs:
    
    # 1.) Observation of Last Action
    oat = atm1
    
    # 2.) Noisy Observation of Current Position
    ot = post + theano_rng.normal((n_o,n_proc))*0.01
    
    # 3.) Nonlinear Transformed Sensory Channel
    oht = T.exp(-T.sqr(post-1.0)/2.0/0.3/0.3)
   
    # Infer hidden state from last hidden state and current observations, using variational density
    hst =  T.nnet.relu( T.dot(Wq_hst_stm1,T.reshape(stm1,(n_s,n_proc))) + T.dot(Wq_hst_ot,T.reshape(ot,(n_o,n_proc))) + T.dot(Wq_hst_oht,T.reshape(oht,(n_oh,n_proc))) + T.dot(Wq_hst_oat,T.reshape(oat,(n_oa,n_proc))) + bq_hst )
    hst2 =  T.nnet.relu( T.dot(Wq_hst2_hst,T.reshape(hst,(n_s,n_proc))) + bq_hst2 )

    stmu =  T.tanh( T.dot(Wq_stmu_hst2,T.reshape(hst2,(n_s,n_proc))) + bq_stmu )
    stsig = T.nnet.softplus( T.dot(Wq_stsig_hst2,T.reshape(hst2,(n_s,n_proc))) + bq_stsig ) + sig_min_states
    
    # Explicitly encode position as homeostatic state variable
    # Rescale representation to fit within linear response of the tanh-nonlinearity
    stmu = T.set_subtensor(stmu[0,:],0.1*ot[0,:]).reshape((n_s,n_proc))
    stsig = T.set_subtensor(stsig[0,:],0.005).reshape((n_s,n_proc))
    
    # Sample from variational density
    st = stmu + theano_rng.normal((n_s,n_proc))*stsig
    
    # Calculate parameters of likelihood distributions from sampled state
    ost = T.nnet.relu( T.dot(Wl_ost_st,T.reshape(st,(n_s,n_proc))) + bl_ost )
    ost2 = T.nnet.relu( T.dot(Wl_ost2_ost,T.reshape(ost,(n_s,n_proc))) + bl_ost2 )
    ost3 = T.nnet.relu( T.dot(Wl_ost3_ost2,T.reshape(ost2,(n_s,n_proc))) + bl_ost3 )
    
    otmu = T.dot(Wl_otmu_st, T.reshape(ost3,(n_s,n_proc))) + bl_otmu
    otsig = T.nnet.softplus(T.dot(Wl_otsig_st, T.reshape(ost3,(n_s,n_proc))) + bl_otsig) + sig_min_obs
    
    ohtmu = T.dot(Wl_ohtmu_st, T.reshape(ost3,(n_s,n_proc))) + bl_ohtmu
    ohtsig = T.nnet.softplus( T.dot(Wl_ohtsig_st, T.reshape(ost3,(n_s,n_proc))) + bl_ohtsig ) + sig_min_obs
    
    oatmu = T.dot(Wl_oatmu_st, T.reshape(ost3,(n_s,n_proc))) + bl_oatmu
    oatsig = T.nnet.softplus( T.dot(Wl_oatsig_st, T.reshape(ost3,(n_s,n_proc))) + bl_oatsig ) + sig_min_obs
    
    # Calculate negative log-likelihood of observations
    p_ot  = GaussianNLL(ot, otmu, otsig)
    p_oht = GaussianNLL(oht, ohtmu, ohtsig)    
    p_oat = GaussianNLL(oat, oatmu, oatsig)
    
    # Calculate prior expectation on hidden state from previous state
    prior_stmu = T.tanh( T.dot(Wl_stmu_stm1, T.reshape(stm1,(n_s,n_proc))) + bl_stmu )
    prior_stsig = T.nnet.softplus( T.dot(Wl_stsig_stm1, T.reshape(stm1,(n_s,n_proc))) + bl_stsig ) + sig_min_states
    
    # Explicitly encode expectations on homeostatic state variable
    prior_stmu = ifelse(T.lt(t,20),prior_stmu, T.set_subtensor(prior_stmu[0,:],0.1))
    prior_stsig = ifelse(T.lt(t,20),prior_stsig, T.set_subtensor(prior_stsig[0,:],0.005))    
   
    # Calculate KL divergence between variational density and prior density
    # using explicit formula for diagonal gaussians
    KL_st = KLGaussianGaussian(stmu, stsig, prior_stmu, prior_stsig)
    
    # Put free energy functional together
    FEt =  KL_st + p_ot + p_oht + p_oat
    
    return st, post, vt, oat, ot, oht, FEt, KL_st, stmu, stsig, force, p_ot, p_oht, p_oat
     
s_t0 = theano.shared(name = 's_t0', value = numpy.zeros( (n_s, n_proc) ).astype( dtype = theano.config.floatX ), borrow = True )
a_t0 = theano.shared(name = 'a_t0', value = numpy.zeros( (n_oa, n_proc) ).astype( dtype = theano.config.floatX ), borrow = True )
o_t0 = theano.shared(name = 'o_t0', value = numpy.zeros( (n_o, n_proc) ).astype( dtype = theano.config.floatX ), borrow = True )
oh_t0 = theano.shared(name = 'oh_t0', value = numpy.zeros( (n_oh, n_proc) ).astype( dtype = theano.config.floatX ), borrow = True )
pos_t0 = theano.shared(name = 'pos_t0', value = -0.5 + 0.0*numpy.random.randn( n_o, n_proc ).astype( dtype = theano.config.floatX ), borrow = True )
v_t0 = theano.shared(name = 'v_t0', value = 0.0*numpy.random.randn( n_o, n_proc ).astype( dtype = theano.config.floatX ), borrow = True )
    
observations_th = T.tensor3('observations_th')
rewards_th = T.tensor3('rewards_th')
action_observations_th = T.tensor3('actions_in_th')
state_noise_th = T.tensor3('state_noise_th')
action_noise_th = T.tensor3('action_noise_th')
    
((states_th, post_th, vt_th, oat_th, ot_th, oht_th, FEt_th, KL_st_th, stmu_th, stsig_th, aht_th, p_ot_th, p_oht_th, p_oat_th), fe_updates) =\
                     theano.scan(fn=inner_fn,
                     sequences = [theano.shared(numpy.arange(n_run_steps).astype(dtype = theano.config.floatX))],
                     outputs_info=[s_t0, pos_t0, v_t0, None, None, None, None, None, None, None, None, None, None, None])
                     
last_stmus = stmu_th[-1,0,:].squeeze().dimshuffle('x',0)
last_stsigs = stsig_th[-1,0,:].squeeze().dimshuffle('x',0)

fixed_prior_cost = GaussianNLL(last_stmus, 0.3, 0.01)
                     
FE_mean = FEt_th.mean()# + fixed_prior_cost.mean()
KL_st_mean = KL_st_th.mean()
p_ot_mean = p_ot_th.mean()
p_oht_mean = p_oht_th.mean()
p_oat_mean = p_oat_th.mean()

run_agent_scan = theano.function(inputs = [], outputs = [states_th, oat_th, ot_th, oht_th, FEt_th, KL_st_th, stmu_th, stsig_th, aht_th], allow_input_downcast = True, on_unused_input='ignore')

#######################################################
#
# Propagate agent through the environment
# and Plot Results
#
#######################################################

states, actions, observations, rewards, FEs, KLs, stmus, stsigs, ahts = run_agent_scan()

fig_propagated = plt.figure(1,figsize=(12,12))
plt.subplot(2,2,2)
plt.title('$o_x(t)$ propagated')
for j in range(10):
    plt.plot(observations[:,0,j].squeeze())
plt.xlabel('t / Steps')
plt.ylabel('$o_x$ / a.u.')
plt.subplot(2,2,3)
plt.title('$o_h(t)$ propagated')
for j in range(10):
    plt.plot(rewards[:,0,j].squeeze())
plt.xlabel('t / Steps')
plt.ylabel('$o_h$ / a.u.')
plt.subplot(2,2,1)
plt.title('$o_a(t)$ propagated')
for j in range(10):
    plt.plot(actions[:,0,j].squeeze())
plt.xlabel('t / Steps')
plt.ylabel('$o_a$ / a.u.')
plt.subplot(2,2,4)
plt.title('$s_1(t)$ propagated')
for j in range(10):
    plt.plot(states[:,0,j].squeeze())
plt.xlabel('t / Steps')
plt.ylabel('$s_1$ / a.u.')

fig_propagated.subplots_adjust(left=0.15, bottom=0.1, right=0.95, top=0.95,
                    wspace=0.3, hspace=0.3)
                    
fig_propagated.savefig('fig_only_proprioceptive_action_propagated.pdf')

# Save mean action to use as template for constrained sampling later
oat_given_mean = actions.mean(axis = 2).squeeze()

########################################################
#
# Test free energy calculation
#
########################################################

free_energy = theano.function([], [FE_mean, KL_st_mean, p_ot_mean, p_oht_mean, p_oat_mean], allow_input_downcast = True, on_unused_input='ignore')

free_energy_sum = free_energy()

print 'Free Energy'
print free_energy_sum

#############################################################
#
# Define Sampling Function for Simulated Run
#
#############################################################

def inner_fn_sample(stm1):
    
    prior_stmu = T.tanh( T.dot(Wl_stmu_stm1, stm1) + bl_stmu )
    prior_stsig = T.nnet.softplus( T.dot(Wl_stsig_stm1, stm1) + bl_stsig ) + sig_min_states
    
    # Set explicit prior on score during last time step
    #prior_stmu = ifelse(T.lt(t,n_run_steps - 5),prior_stmu, T.set_subtensor(prior_stmu[0,:],0.1))
    #prior_stsig = ifelse(T.lt(t,n_run_steps - 5),prior_stsig, T.set_subtensor(prior_stsig[0,:],0.001))    
    
    st = prior_stmu + theano_rng.normal((n_s,n_samples))*prior_stsig
    
    ost = T.nnet.relu( T.dot(Wl_ost_st,st) + bl_ost )
    ost2 = T.nnet.relu( T.dot(Wl_ost2_ost,ost) + bl_ost2 )
    ost3 = T.nnet.relu( T.dot(Wl_ost3_ost2,ost2) + bl_ost3 )
    
    otmu = T.dot(Wl_otmu_st, ost3) + bl_otmu
    otsig = T.nnet.softplus(T.dot(Wl_otsig_st, ost3) + bl_otsig) + sig_min_obs
    
    ohtmu = T.dot(Wl_ohtmu_st, ost3) + bl_ohtmu
    ohtsig = T.nnet.softplus( T.dot(Wl_ohtsig_st, ost3) + bl_ohtsig ) + sig_min_obs
    
    oatmu = T.dot(Wl_oatmu_st, ost3) + bl_oatmu
    oatsig = T.nnet.softplus( T.dot(Wl_oatsig_st, ost3) + bl_oatsig ) + sig_min_obs
    
    ot = otmu + theano_rng.normal((n_o,n_samples))*otsig
    oht = ohtmu + theano_rng.normal((n_oh,n_samples))*ohtsig   
    oat = oatmu + theano_rng.normal((n_oa,n_samples))*oatsig   
    
    return st, ohtmu, ohtsig, ot, oht, oat, prior_stmu, prior_stsig
    
# Define initial state and action
    
s_t0_sample = theano.shared(name = 's_t0', value = numpy.zeros( (n_s,n_samples) ).astype( dtype = theano.config.floatX ), borrow = True )
    

    
((states_sampled, reward_probabilities_mu_sampled, reward_probabilities_sig_sampled, observations_sampled, rewards_sampled, actions_observations_sampled, stmus_sampled, stsigs_sampled), updates_sampling) =\
                     theano.scan(fn=inner_fn_sample,
                     outputs_info=[s_t0_sample, None, None, None, None, None, None, None],
                     n_steps = n_sample_steps)                     

########################################################
#
# Run and Plot Results for Free Sampling
#
########################################################

# Build Function
eval_Penalized_FE = theano.function([], [FE_mean, observations_sampled, rewards_sampled, actions_observations_sampled, states_sampled], allow_input_downcast = True, on_unused_input='ignore')

# Evaluate Function
#print 'Penalized FE'
results_eval =  eval_Penalized_FE()
#print results_eval

FE_min = results_eval[0]

observations = results_eval[1]
rewards = results_eval[2]
actions = results_eval[3]
states = results_eval[4]

print 'FE_min:'
print FE_min

fig_sampled = plt.figure(2,figsize=(12,12))
plt.subplot(2,2,2)
plt.title('$o_x(t)$ sampled')
for j in range(10):
    plt.plot(observations[:,0,j].squeeze())
plt.xlabel('t / Steps')
plt.ylabel('$o_x$ / a.u.')
plt.subplot(2,2,3)
plt.title('$o_h(t)$ sampled')
for j in range(10):
    plt.plot(rewards[:,0,j].squeeze())
plt.xlabel('t / Steps')
plt.ylabel('$o_h$ / a.u.')
plt.subplot(2,2,1)
plt.title('$o_a(t)$ sampled')
for j in range(10):
    plt.plot(actions[:,0,j].squeeze())
plt.xlabel('t / Steps')
plt.ylabel('$o_a$ / a.u.')
plt.subplot(2,2,4)
plt.title('$s_1(t)$ sampled')
for j in range(10):
    plt.plot(states[:,0,j].squeeze())
plt.xlabel('t / Steps')
plt.ylabel('$s_1$ / a.u.')
fig_sampled.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95,
                    wspace=0.3, hspace=0.3)
fig_sampled.savefig('fig_only_proprioceptive_action_sampled.pdf')

plt.show()
