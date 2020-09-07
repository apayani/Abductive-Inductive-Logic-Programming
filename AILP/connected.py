"""
Created on Mon Jul 27 20:50:55 2020

@author: Ali Payani
"""

import os, logging

os.environ["DNLILP_OR_TYPE"] = "expsumlog"
os.environ["DNLILP_AND_TYPE"] = "expsumlog"

from Lib.ILPCore import ILPEngine
from Lib.predCollection import PredCollection
from Lib.BackgroundType1 import  BackgroundType1
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

    

ITER=1000
params=DotDict()
params.T=5
params.BS=None
params.LR=.01
params.SEED=0

#define constants
C=['a','b','c','d','e'] 
Constants = dict({'C': C,'D':[1,2,3,4] })

predColl = PredCollection (Constants)
 
#define predicates
predColl.add_pred( name='edge'  ,arguments=['C','C'])

def Fn():
    return DNFLayer( [10,1],sig=1., and_init=[-1,.1],or_init=[-1,.1] )

cnt  = predColl.add_pred(name='connected',arguments=['C','C'],  Fam= LOP.or2_op()) \
    .add_rule( variables=['C'  ],  Fn= Fn, use_neg=False, exc_preds=[]) 
 
bg = BackgroundType1( predColl )



bg.constants['C']=C
bg.add_backgroud ('edge', ('a','b') ) 
bg.add_backgroud ('edge', ('b','c') )
bg.add_backgroud ('edge', ('c','d') )
bg.add_backgroud ('edge', ('e','d') )

for p in predColl.preds:
    if p.rules:
        bg.add_zero_backgroud(p.name)        

bg.add_example('connected',('a','b'))
bg.add_example('connected',('a','c'))
bg.add_example('connected',('a','d'))
bg.add_example('connected',('b','c'))
bg.add_example('connected',('b','d'))
bg.add_example('connected',('c','d'))
bg.add_example('connected',('e','d'))


predColl.initialize_predicates() 
bgc=bg.compile_bg()

###########################################################################

model = ILPEngine( params=params ,predColl=predColl  )
import time 

st=time.time()
tr_fn,te_fn=model.define_model( ['connected'],['connected'],{'':tf.keras.optimizers.Adam(params.LR )})

for i in range(1000000):
    loss = tr_fn (  [bgc] )  
    if i%100==0:
        print('i = %d, loss=%.3f'%(i,loss))
    
    if loss<1e-3:
        print('optimized !')
        break

en=time.time()
print('elapsed time : ' , en-st)
#print( np.round(100* te_fn ( bgc)[0]['connected'])/100.   )

model.ForwardChain.fn_dict [str(cnt.rules[0])].print(1)

 