

#%%
import os, logging

os.environ["DNLILP_OR_TYPE"] = "expsumlog"
os.environ["DNLILP_AND_TYPE"] = "expsumlog"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)



from Lib.ILPCore import ILPEngine
from Lib.PredicateLib import PredCollection
from Lib.logicLayers import *
from Lib.logicOps import LOP
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd 
import pickle
from time import sleep
from datetime import datetime
from sklearn import metrics 
from functools import  partial
from sklearn.model_selection import train_test_split
db=pd.read_csv('sonar.txt')

X=db.iloc[:,:-1]
Y=(db.iloc[:,-1]=='M')*1.0
X=X.to_numpy()
X=X-X.min(axis=0,keepdims=True)
X=X/X.max(axis=0,keepdims=True)
Y=Y.to_numpy()

Xtrc, Xtec, Ytr, Yte= train_test_split(X, Y, test_size=0.2, random_state=42)
 


ITER=1000
params=DotDict()
params.T=1
params.BS=None
params.LR=.01
params.SEED=0

# def maj_op(x,axis=-1,keepdims=False):
#     S = .5 + (tf.math.reduce_sum(x,axis,keepdims=keepdims)-.5)/x.shape[-1]
#     return relu1(S)
# def xor_op(x,axis=-1,keepdims=False):
#     S = tf.math.reduce_prod(2*x-1,axis,keepdims=keepdims)
#     return .5 +S*.5

  
 
print('continuous', Xtrc.shape)

predColl = PredCollection ( { })
 

predColl.add_pred(name='false')
preds=[]
    
predColl.add_continous_vars( 'cnt',Xtrc.shape[-1])
 

 
for i in range(1):
   
    pred = predColl.add_pred(name='m',Fam= LOP.eq2_op(), Frules= LOP.or_op(),target=True)\
        .add_rule( 
        pFunc=DNFLayer ( dims = [5  ,1], sig=3., and_init=[-3,.2], or_init=[-3,.1] ,L1=0.001,L2=0.05,or_count=10,max_term=20,randmaskpercent=0.,cnf=False,negate=False),
        convertors=[    ( 'cnt',CNTLayer('nN',N=1)) ],
        use_neg=False,  Fout= lambda x: x)
    
  
predColl.initialize_predicates()    
model = ILPEngine( params=params ,predColl=predColl  )


train_fn={}
 
opts={}
opts['logic_layer'] = tf.keras.optimizers.Adam(.01 )
opts['cnt_layer'] = tf.keras.optimizers.Adam(.001 )
train_fn['m'] = model.define_model_train_filt2( ['m'],opts)
test_fn =  model.define_model_test(['m'],['m'])



max_acc=0
 
padL=0
for p in predColl.preds:
    if p.rules:
        padL+= p.pairs_len
X_tr = dict()
X_te =dict()
for p in predColl.preds:
    X_tr['x0_' + p.name] = np.zeros( (Xtrc.shape[0],p.pairs_len))
    X_te['x0_' + p.name] = np.zeros( (Xtec.shape[0],p.pairs_len))
 
X_tr['cnt_cnt']= Xtrc
 
X_te['cnt_cnt']= Xtec
 
label_tr={}
label_te={}

pg = {}
for p in preds+['m']:
    label_tr[p] =  Ytr[:,np.newaxis]
    label_te[p] =  Yte[:,np.newaxis]
    pg[p] = .003 
pg['m']=.5
  
#%%
def getth(predicted,target):
    fpr, tpr, threshold = roc_curve(target, predicted)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    return optimal_threshold
last_err=0
index=5
 
 
for epoch in range(ITER):

    p='m'
    print('-----------------------------------------------------------------------------')
    print('epoch = ', epoch, 'max acc', max_acc)

    BS=15
    
    # xo_tr,   loss_tr  = model.process_cl_dataset_gen ( is_train=True,  fn=fn, X=X_tr ,labels=label_tr, target_names=[p], batch_size=BS, plogent=pg, randomize=True)
    loss_tr  = model.process_cl_dataset_train (  train_fn[p] , X=X_tr ,labels=label_tr, target_names=[p], batch_size=BS, plogent=pg, randomize=True)
    
    BS=3
    xo_tr,   loss_tr  = model.process_cl_dataset_gen ( is_train=False, fn=test_fn, X=X_tr ,labels=label_tr, target_names= [p], batch_size=BS, plogent=pg, randomize=False)   
    xo_te,   loss_te  = model.process_cl_dataset_gen ( is_train=False, fn=test_fn, X=X_te ,labels=label_te, target_names= [p], batch_size=BS, plogent=pg, randomize=False)   
    
    sleep(1)
    auc_score,aupr_score = get_auc(Yte,xo_te[p][:,0] )
    max_acc = max( auc_score,max_acc)
    
    print( np.mean(  Ytr!= ( xo_tr[p][:,0]> getth(xo_tr[p][:,0], Ytr))),  np.mean(  Yte!= ( xo_te[p][:,0]> getth(xo_te[p][:,0], Yte))))
    err = np.mean(  Yte!= (xo_te[p][:,0]>.5))
    
    print('\nlosses:', loss_tr,loss_te,err)
    # print('\nTraining.... ',epoch,p,'avg loss = ', np.mean(loss_tr),  get_auc(Ytr,xo_tr[p]) )
    print('Testing.... ',p,'auc : %.8f,  aupr :   %.8f'%(auc_score,aupr_score ) )
         


# %%
