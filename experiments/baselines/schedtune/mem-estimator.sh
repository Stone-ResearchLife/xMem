#!/bin/bash
#Hadeel Albahar


#Input: predicted GPU memory (P), Maximum prediction error (E), and GPU
#Output: Estimated GPU memory or OOM Error

#if ! [[ $# -eq 4 ]] 
#then
#   echo "Usage: $0 <PREDICTED-MEM> <MAX-ERROR> <GPU> <optional flag: v>"
#   echo 
#   echo
#   echo "Example: $0 <pred-mem> <max-error> <avail-mem> 1 v"
#   echo "Example: $0 <pred-mem> <max-error> <avail-mem> 3"
#   exit 1
#fi


pred_mem=$1
max_error=$2
error=$(echo "$max_error / 100" | bc -l)
#echo "$error"
error_mb=$(echo "$pred_mem * $error" | bc)
#echo "$error_mb"
estimate=$(echo "$pred_mem + $error_mb" | bc)
#echo "$estimate"

#if GPU is either 2070s or 3070
if [[ $3 == "1" ]] || [[ $3 == "2" ]]
then
   #if we need to check if it can even fit alone
   if [[ $4 == "v" ]]
   then
      if (( $(echo "$estimate > 7982" | bc -l) )) 
      then
         echo "INVALID"
      else
         echo "VALID"
      fi
   else
      echo "$estimate"
      #if (( $(echo "$estimate > $avail_gpu_mem" | bc -l) ))
      #then
      #   echo "OOM"
      #else
      #   echo "$estimate"
      #fi
   fi

else
   #if we need to check if it can even fit alone
   if [[ $4 == "v" ]]
   then
      if (( $(echo "$estimate > 24250" | bc -l) ))
      then
         echo "INVALID"
      else
         echo "VALID"
      fi
   else
      echo "$estimate"
      #if (( $(echo "$estimate > $avail_gpu_mem" | bc -l) ))
      #then
      #   echo "OOM"
      #else
      #   echo "$estimate"
      #fi
   fi
fi
