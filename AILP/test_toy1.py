#%%
import os, logging
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

import numpy as np
import pandas as pd 
import pickle
from time import sleep
from datetime import datetime
from sklearn import metrics 
from functools import  partial
 
    
ITER=1000
params=DotDict()
params.T=1
params.BS=None
params.LR=.01
params.SEED=0


 
import numpy as np
import pandas as pd 
import argparse

 
from time import sleep
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd, numpy as np
from sklearn import preprocessing
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report, confusion_matrix, f1_score, roc_curve, roc_auc_score
import pandas as pd
import copy        
from sklearn import metrics 
from functools import  partial
import time
 
Ntr=100000
Nte=20000

Xtrb=np.random.randint( low=0,high=2, size=(Ntr,4))
Xteb=np.random.randint( low=0,high=2, size=(Nte,4))

Xtrc=np.random.rand(Ntr,4)
Xtec=np.random.rand(Nte,4)


def func23(b,c):
    
    cond1 = np.logical_and(c[:,0]>.53,c[:,0]<.55 ) 
    cond2 = np.logical_and(c[:,0]>.23,c[:,0]<.35 ) 
    cond3 = np.logical_and(c[:,0]>.73,c[:,0]<.85 ) 
    cond=np.logical_or( cond1,cond2)
    cond=np.logical_or( cond,cond3)
    x1 = b[:,0]*(1.0-b[:,1]) *  cond
    x2 = b[:,0]*b[:,1] *  (1.0-cond)
    x1 = np.logical_or(x1,x2)
    return x1

def func2(b,c):
    
    cond1 = np.logical_and(c[:,0]>.53,c[:,0]<.55 ) 
    cond2 = np.logical_and(c[:,0]>.23,c[:,0]<.35 ) 
     
    cond=np.logical_or( cond1,cond2)
    
    x1 = b[:,0]*(1.0-b[:,1]) *  cond
    x2 = b[:,0]*b[:,1] *  (1.0-cond)
    x1 = np.logical_or(x1,x2)
    return x1
    
def func(b,c):
    cond = np.logical_and(c[:,0]>.53,c[:,0]<.55 ) 
    x1 = b[:,0]*(1.0-b[:,1]) *  cond
    x2 = b[:,0]*b[:,1] *  (1.0-cond)
    x1 = np.logical_or(x1,x2)
    return x1



def func1(b,c):
    
 
    return c[:,0]>=.55
    

Ytr = func( Xtrb,Xtrc)
Yte = func( Xteb,Xtec)
print( np.sum(Yte))
print( np.sum(Ytr))


 
print('binaries', Xtrb.shape)
print('continuous', Xtrc.shape)

 
predColl = PredCollection ( { })


bins =  CNTLayer('bin',neg=True) 

predColl.add_pred(name='false')
preds=[]

predColl.add_continous_vars( 'bin',Xtrb.shape[-1])
predColl.add_continous_vars( 'cnt',Xtrc.shape[-1])
 
 
predColl.add_pred(name='false')
 
cnt_i = CNTLayer('nN',neg=True,N=2) 
pFunc = DNFLayer ( dims = [10 ,1], sig=1., and_init=[-2,1], or_init=[-1,.1],or_count=0 )
convertors= [ ('bin',bins),('cnt',cnt_i)]
   
Fout = lambda x:x
predColl.add_pred(name='m',arguments=[], target=True, Fam=LOP.eq2_op(), Frules=LOP.or_op()) \
        .add_rule( pFunc=pFunc,convertors=convertors,use_neg=False, inc_preds= ['false'], Fout= Fout) \
   
      
predColl.initialize_predicates()    
model = ILPEngine( params=params,predColl=predColl  )

# tr_fn,te_fn=model.define_model_bin( ['m'])
# tr_fn = model.define_model_train_filt2( ['m'], {'': tf.keras.optimizers.Adam(.01 )} )
opts={}
opts['logic_layer'] = tf.keras.optimizers.Adam(.01 )
opts['cnt_layer'] = tf.keras.optimizers.Adam(.001 )

tr_fn = model.define_model_train_filt2(['m'], opts )
te_fn = model.define_model_test(['m'], ['m'] )


max_acc=0
 
 
 
 
padL=0
for p in predColl.preds:
    if p.rules:
        padL+= p.pairs_len
X_tr = dict()
X_te =dict()
for p in predColl.preds:
    X_tr['x0_' + p.name] = np.zeros( (Xtrb.shape[0],p.pairs_len))
    X_te['x0_' + p.name] = np.zeros( (Xteb.shape[0],p.pairs_len))
X_tr['cnt_bin']= Xtrb
X_tr['cnt_cnt']= Xtrc
X_te['cnt_bin']= Xteb
X_te['cnt_cnt']= Xtec
label_tr={}
label_te={}
 

for p in preds+['m']:
    label_tr[p] =  Ytr[:,np.newaxis]
    label_te[p] =  Yte[:,np.newaxis]

BS=1000
import time
 
 
last_err=0
 
index=5
pg={}
pg['m'] = .5
 
import time

for epoch in range(100000):

      
    BS=100 
    p='m'
    loss_tr  = model.process_cl_dataset_train(  fn=tr_fn, X=X_tr ,labels=label_tr, target_names=['m'], batch_size=BS, plogent=pg, randomize=False)
    BS=1000
    xo,loss_tr  = model.process_cl_dataset_gen( is_train=False,  fn=te_fn, X=X_tr ,labels=label_tr, target_names=['m'], batch_size=BS, plogent=pg, randomize=False)
    
 
    
    print('\n', epoch, loss_tr, np.mean( abs(xo['m']-label_tr[p]))  )
   
       
    
