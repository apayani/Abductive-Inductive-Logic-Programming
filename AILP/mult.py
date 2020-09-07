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
params.T=8
params.BS=None
params.LR=.001
params.SEED=0

maxN=7
#define constants
N=[ '%d'%i for i in range(maxN+1)]
Constants = {'N':N}
 
predColl = PredCollection (Constants)

 
#define predicates
 

predColl.add_pred( name='zero'  ,arguments=['N'])
predColl.add_pred( name='inc'  ,arguments=['N','N'])
predColl.add_pred( name='add'  ,arguments=['N','N','N'])

 
def Fn():
    return DNFLayer( [8,1],sig=2., and_init=[-3,.5],or_init=[-3,1] )

 

cnt = predColl.add_pred(name='mul',arguments=['N','N','N'], Fam= tf.maximum) \
     .add_rule( variables=['N' ,'N' ],  Fn= Fn, use_neg=False, exc_preds=[], exc_conds=[('*','rep1') ],Fvar=LOP.or_op_max) 
    
     
   
predColl.initialize_predicates()    


 
#add background
bg = BackgroundType1( predColl )
bg.constants['N']=N
bg.add_backgroud ('zero', ('0',) ) 
for i in range(maxN+1):
    if i+1 in N:
        bg.add_backgroud ('inc', (i,i+1) ) 
    for j in range(maxN+1):
        if '%d'%(i+j) in N:
            bg.add_backgroud ('add', ('%d'%i,'%d'%i,'%d'%(i+j)) ) 
        if '%d'%(i*j) in N:
            bg.add_example ('mul',('%d'%i,'%d'%i,'%d'%(i*j)) ) 
            
for p in predColl.preds:
    if p.rules:
        bg.add_zero_backgroud(p.name)        


predColl.initialize_predicates() 
bgc=bg.compile_bg()

###########################################################################

model = ILPEngine( params=params ,predColl=predColl  )
import time 

st=time.time()
tr_fn,te_fn=model.define_model( ['mul'],['mul'],{'':tf.keras.optimizers.Adam(params.LR )})

for i in range(1000000):
    loss = tr_fn (  [bgc] )  
    if i%20 ==0:
        print('i = %d, loss=%.3f'%(i,loss))
        model.ForwardChain.fn_dict [str(cnt.rules[0])].print(1)
    
    if loss<1e-3:
        print('optimized !')
        break
    


en=time.time()
print('elapsed time : ' , en-st)
#print( np.round(100* te_fn ( bgc)[0]['connected'])/100.   )



 
 