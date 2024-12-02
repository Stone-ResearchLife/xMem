#!/usr/bin/env python3.6
#Hadeel Albahar
import os
import sys
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Keras outputs warnings using `print` to stderr so let's direct that to devnull temporarily
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
'''
./predictor.py -j $job_name -o $1 -a $activations -p $parameters -i $input -g $gpu_2070s
'''

import argparse
import joblib
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings("ignore")

#GPUs in the cluster
#CUDA_cores,MemoryBW_GBps,Memory_clock_speed_MHz,Tensor_cores,SM_count
gpu_2070s=[2560,448,14000,320,40]
gpu_3070=[5888,512,16000,184,46]
gpu_3080=[8704,760.3,19000,272,68]
gpu_3090=[5120,900,1752,640,80]


parser = argparse.ArgumentParser(description='Loading predictors and returning prediction.')
parser.add_argument('-j', '--jobname', help='Job name')
parser.add_argument('-o', '--option', help='option either 1 or 2')
parser.add_argument('-a', '--activations', help='Model activations')
parser.add_argument('-p', '--parameters', help='Model parameters')
parser.add_argument('-i', '--inputsize', help='Model input size')
parser.add_argument('-g', '--gpu', help='GPU either 2070s or 3070 or 3090')
args = parser.parse_args()

'''
print(args.jobname)
print(args.option)
print(args.activations)
print(args.parameters)
print(args.inputsize)
print(args.gpu)
'''

if args.gpu == "2070s":
    gpu = gpu_2070s
elif args.gpu == "3070":
    gpu = gpu_3070
elif args.gpu == "3080":
    gpu = gpu_3080
elif args.gpu == "3090":
    gpu = gpu_3090

if args.option == "1":
    #print("option 1")
    if "batchsize" in args.jobname:
        #print("train job")
        train_mem_pred=joblib.load("./train_mem_RFR_3params.joblib")
        ##CUDA_cores,MemoryBW_GBps,Memory_clock_speed_MHz,Tensor_cores,SM_count
        #X = df[['activations', 'parameters','input','Memory bandwidth (GB/s)','Pipelines/CUDA cores', 'SM count']]
        tm = train_mem_pred.predict([[float(args.activations),float(args.parameters),float(args.inputsize),gpu[1],gpu[0],gpu[4]]])
        if tm.shape == (1,):
            tain_mem = tm[0]
        elif tm.shape == (1,1):
            tain_mem = tm[0,0]
        train_time_pred=joblib.load("train_time_RFR_5params.joblib")
        ##CUDA_cores,MemoryBW_GBps,Memory_clock_speed_MHz,Tensor_cores,SM_count
        #X = df[['activations', 'parameters','input','Pipelines/CUDA cores', 'Memory bandwidth (GB/s)', 'SM count','Memory clock speed (MHz)','Tensor cores']]
        tt = train_time_pred.predict([[float(args.activations),float(args.parameters),float(args.inputsize),gpu[0],gpu[1],gpu[4],gpu[2],gpu[3]]])
        if tt.shape == (1,):
            train_time = tt[0]
        elif tt.shape == (1,1):
            train_time = tt[0,0]
        print("%f,%f" % (tain_mem,train_time))

    ##inference-vgg19-cat2
    if "inference" in args.jobname:
        infer_mem_pred=joblib.load("infer_mem_RFR_5params.joblib")
        #X = df[['activations', 'parameters', 'Memory bandwidth (GB/s)','Pipelines/CUDA cores', 'SM count', 'Memory clock speed (MHz)','Tensor Cores (GPU)']]
        im = infer_mem_pred.predict([[float(args.activations),float(args.parameters),gpu[1],gpu[0],gpu[4],gpu[2],gpu[3]]])
        if im.shape == (1,):
            infer_mem = im[0]
        elif im.shape == (1,1):
            infer_mem = im[0,0]

        infer_time_pred=joblib.load("infer_time_RFR_5params.joblib")
        it = infer_time_pred.predict([[float(args.activations),float(args.parameters),gpu[1],gpu[0],gpu[4],gpu[2],gpu[3]]])
        if it.shape == (1,):
            infer_time = it[0]
        elif it.shape == (1,1):
            infer_time = it[0,0]
        print("%f,%f" % (infer_mem,infer_time))


elif args.option == "2":
    if "batchsize" in args.jobname:
        train_mem_pred=joblib.load("train_mem_RFR_5params.joblib")
        ##CUDA_cores,MemoryBW_GBps,Memory_clock_speed_MHz,Tensor_cores,SM_count
        #X = df[['activations', 'parameters','input', 'Memory bandwidth (GB/s)', 'Pipelines/CUDA cores','SM count','Memory clock speed (MHz)','Tensor cores']]
        tm = train_mem_pred.predict([[float(args.activations),float(args.parameters),float(args.inputsize),gpu[1],gpu[0],gpu[4],gpu[2],gpu[3]]])
        if tm.shape == (1,):
            tain_mem = tm[0]
        elif tm.shape == (1,1):
            tain_mem = tm[0,0]
        train_time_pred=joblib.load("train_time_RFR_5params.joblib")
        ##CUDA_cores,MemoryBW_GBps,Memory_clock_speed_MHz,Tensor_cores,SM_count
        #X = df[['activations', 'parameters','input','Pipelines/CUDA cores', 'Memory bandwidth (GB/s)', 'SM count','Memory clock speed (MHz)','Tensor cores']]
        tt = train_time_pred.predict([[float(args.activations),float(args.parameters),float(args.inputsize),gpu[0],gpu[1],gpu[4],gpu[2],gpu[3]]])
        if tt.shape == (1,):
            train_time = tt[0]
        elif tt.shape == (1,1):
            train_time = tt[0,0]
        print("%f,%f" % (tain_mem,train_time))


    if "inference" in args.jobname:
        infer_mem_pred=joblib.load("infer_mem_RFR_3params.joblib")
        #X = df[['activations', 'parameters', 'Memory bandwidth (GB/s)','Pipelines/CUDA cores', 'SM count']]
        im = infer_mem_pred.predict([[float(args.activations),float(args.parameters),gpu[1],gpu[0],gpu[4]]])
        if im.shape == (1,):
            infer_mem = im[0]
        elif im.shape == (1,1):
            infer_mem = im[0,0]

        infer_time_pred=joblib.load("infer_time_RFR_3params.joblib")
        #X = df[['activations', 'parameters', 'Memory bandwidth (GB/s)','Pipelines/CUDA cores', 'SM count']]
        it = infer_time_pred.predict([[float(args.activations),float(args.parameters),gpu[1],gpu[0],gpu[4]]])
        if it.shape == (1,):
            infer_time = it[0]
        elif it.shape == (1,1):
            infer_time = it[0,0]
        print("%f,%f" % (infer_mem,infer_time))
