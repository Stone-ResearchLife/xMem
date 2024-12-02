#!/bin/bash
#Hadeel Albahar

./monitor_gpus.sh

touch delete_commands.sh
echo "#!/bin/bash" > delete_commands.sh
#GPUs in the cluster
#CUDA_cores,MemoryBW_GBps,Memory_clock_speed_MHz,Tensor_cores,SM_count
#gpu_2070s="2560,448,14000,320,40"
#gpu_3070="5888,512,16000,184,46"
#gpu_3090="5120,900,1752,640,80"

if [ $# -eq 1 ]; then
    option=$1
else
   echo "Usage: $0 <option number>"
   echo "Example: $0 1, for train_mem_RFR_3params.joblib, train_time_neuralnet.joblib, infer_mem_RFR_5params.joblib, infer_time_RFR_3params.joblib"
   echo "Example: $0 2, for train_mem_RFR_5params.joblib, train_time_neuralnet.joblib, infer_mem_RFR_3params.joblib, infer_time_RFR_5params.joblib"
   exit 1
fi

function displaytime {
  local T=$1
  local D=$((T/60/60/24))
  local H=$((T/60/60%24))
  local M=$((T/60%60))
  local S=$((T%60))
  (( $D > 0 )) && printf '%d days ' $D
  (( $H > 0 )) && printf '%d hours ' $H
  (( $M > 0 )) && printf '%d minutes ' $M
  (( $D > 0 || $H > 0 || $M > 0 )) && printf 'and '
  printf '%d seconds\n' $S
}
#####################################################################
STARTTIME=$(date +%s)
echo "$(date) controller: " | tee ./log_ours.txt
echo "controller: Starting ours evaluation..." | tee -a ./log_ours.txt

RED='\e[31m'
NC='\e[0m' # No Color

ext="-job-0"
gpu_2070s="2070s"
gpu_3070="3070"
gpu_3090="3090"

#touch queue.csv
#./scheduler.sh $1 & 

#using the same arrival times as the baseline for fair comparison
input="baseline_arrival_times.csv"
#echo "$(date) controller:      Reading the $input file" | tee -a ./log_ours.txt
while IFS=, read -r datetime job_name job_yaml arrivaltime
do
   #echo $datetime
   #echo $job_name
   #cat $job_yaml
   #echo $arrivaltime
#   sleep $arrivaltime
   echo "$(date) controller:	Next job to run: $job_yaml" | tee -a ./log_ours.txt
   if [[ $job_yaml == job_config_single* ]]
   then
     #single-node-vgg19-cifar10-epochs1-batchsize64
#     echo "$(date) controller:	extracting training job: $job_name information..." | tee -a ./log_ours.txt
     model=$(echo $job_name | cut -d'-' -f3) ; echo "$model"
     bs=$(echo $job_name | cut -d'-' -f6 | grep -Eo '[0-9]{1,4}') ; echo "$bs"
     activations=$(cat ./train/model_stats.csv | grep "$model,$bs" | head -1 | cut -d',' -f3) ; echo "$activations"
     parameters=$(cat ./train/model_stats.csv | grep "$model,$bs" | head -1 | cut -d',' -f4) ; echo "$parameters"
     input=$(cat ./train/model_stats.csv | grep "$model,$bs" | head -1 | cut -d',' -f5) ; echo "$input"
#     echo "$(date) controller:	Getting predictions..." | tee -a ./log_ours.txt
     python3 ./predictor.py -j $job_name -o $1 -a $activations -p $parameters -i $input -g $gpu_2070s > gpu2070s_predictions & pid1=$! 
     python3 ./predictor.py -j $job_name -o $1 -a $activations -p $parameters -i $input -g $gpu_3070 > gpu3070_predictions & pid2=$!
     python3 ./predictor.py -j $job_name -o $1 -a $activations -p $parameters -i $input -g $gpu_3090 > gpu3090_predictions & pid3=$!
     wait "$pid1" "$pid2" "$pid3"
#     echo "$(date) controller: done extracting info and getting predictions for $job_name" | tee -a ./log_ours.txt 
   else
     #inference-vgg19-cat2
#     echo "$(date) controller: extracting inference job: $job_name information..." | tee -a ./log_ours.txt
     model=$(echo $job_name | cut -d'-' -f2)
     activations=$(cat ./train/model_stats.csv | grep "$model" | head -1 | cut -d',' -f3)
     parameters=$(cat ./train/model_stats.csv | grep "$model" | head -1 | cut -d',' -f4)
#     echo "$(date) controller:Getting predictions..." | tee -a ./log_ours.txt
     python3 ./predictor.py -j $job_name -o $1 -a $activations -p $parameters -g $gpu_2070s > gpu2070s_predictions & pid1=$! 
     python3 ./predictor.py -j $job_name -o $1 -a $activations -p $parameters -g $gpu_3070 > gpu3070_predictions & pid2=$!
     python3 ./predictor.py -j $job_name -o $1 -a $activations -p $parameters -g $gpu_3090 > gpu3090_predictions & pid3=$!
     wait "$pid1" "$pid2" "$pid3"
#     echo "$(date) controller: done extracting info and getting predictions for $job_name" | tee -a ./log_ours.txt
   fi
#   echo "$(date) controller: Passing job: $job_name (with yaml: $job_yaml) and predictions to scheduler.sh script..." | tee -a ./log_ours.txt
   #echo "$job_name,$job_yaml,$gpu2070s_mem_pred,$gpu2070s_time_pred,$gpu3070_mem_pred,$gpu3070_time_pred,$gpu3090_mem_pred,$gpu3090_time_pred"
   job_info=$(./enqueue.sh $job_name $job_yaml gpu2070s_predictions gpu3070_predictions gpu3090_predictions)
#   echo "$job_info"
#   echo "Now calling scheduler"
   ./scheduler.sh $1 $job_info
   #echo "enqueue $job_name ... DONE"
   #echo "current jobs in queue.csv: "
   #cat queue.csv
#   echo
#   echo
#   echo "#############################################################"
#   echo
done < "$input"

ENDTIME=$(date +%s)
echo "$(date)" | tee -a ./log_ours.txt
printf "Total time to run all the ours experiments: " | tee -a ./log_ours.txt
displaytime $[$ENDTIME - $STARTTIME] | tee -a ./log_ours.txt
