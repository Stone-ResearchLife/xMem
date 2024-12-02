import os
import joblib
import warnings
from typing import Union, Optional
from pathlib import Path
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



#
#
# parser = argparse.ArgumentParser(description='Loading predictors and returning prediction.')
# parser.add_argument('-j', '--jobname', help='Job name')
# parser.add_argument('-o', '--option', help='option either 1 or 2')
# parser.add_argument('-a', '--activations', help='Model activations')
# parser.add_argument('-p', '--parameters', help='Model parameters')
# parser.add_argument('-i', '--inputsize', help='Model input size')
# parser.add_argument('-g', '--gpu', help='GPU either 2070s or 3070 or 3090')
# ## ======================= New Code =======================
# parser.add_argument('-d', '--conf-dir', type=str, help='Configuration directory', default=None)
# parser.add_argument('--json', type=str, help='JSON file path', default=None)
# ## ======================= End of New Code =======================
# args = parser.parse_args()
#
# '''
# print(args.jobname)
# print(args.option)
# print(args.activations)
# print(args.parameters)
# print(args.inputsize)
# print(args.gpu)
# '''


class Schedtune:
    def __init__(
            self,
            jobname: str,
            option: str,
            activations: Union[int, float],
            parameters: Union[int, float],
            inputsize: Union[int, float],
            gpu: str,
            conf_dir: Optional[Union[Path, str]] = None,
            json: Optional[Union[Path, str]] = None
    ):
        self.jobname = jobname
        self.option = option
        self.activations = activations # in MB
        self.parameters = parameters # in MB
        self.inputsize = inputsize # in MB
        self.gpu = self._get_gpu(gpu)
        self.conf_dir = conf_dir or Path(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'schedtune'))
        if not isinstance(self.conf_dir, Path):
            self.conf_dir: Path = Path(self.conf_dir)
        self.json = json
        if not isinstance(self.json, Path) and self.json is not None:
            self.json: Path = Path(self.json)

    def _get_gpu(self, gpu: str) -> list:
        #GPUs in the cluster
        #CUDA_cores,MemoryBW_GBps,Memory_clock_speed_MHz,Tensor_cores,SM_count
        gpu_2070s=[2560,448,14000,320,40]
        gpu_3070=[5888,512,16000,184,46]
        gpu_3080=[8704,760.3,19000,272,68]
        gpu_3090=[5120,900,1752,640,80]
        gpu_4070ti=[7680,504,21000,240,60]
        gpu_4060=[3072,272,17000,96,24]

        _gpu = None
        if gpu == "2070s":
            _gpu = gpu_2070s
        elif gpu == "3070":
            _gpu = gpu_3070
        elif gpu == "3080":
            _gpu = gpu_3080
        elif gpu == "3090":
            _gpu = gpu_3090
        elif gpu == "4070ti":
            _gpu = gpu_4070ti
        elif gpu == "4060":
            _gpu = gpu_4060
        return _gpu

    def estimate(self):
        output = {
            'mem': None,
            'time': None
        }
        conf_dir = self.conf_dir
        gpu = self.gpu
        if self.option == "1":
            #print("option 1")
            if "batchsize" in self.jobname:
                #print("train job")
                train_mem_pred=joblib.load(conf_dir.joinpath("train_mem_RFR_3params.joblib"))
                ##CUDA_cores,MemoryBW_GBps,Memory_clock_speed_MHz,Tensor_cores,SM_count
                #X = df[['activations', 'parameters','input','Memory bandwidth (GB/s)','Pipelines/CUDA cores', 'SM count']]
                tm = train_mem_pred.predict([[float(self.activations),float(self.parameters),float(self.inputsize),gpu[1],gpu[0],gpu[4]]])
                if tm.shape == (1,):
                    tain_mem = tm[0]
                elif tm.shape == (1,1):
                    tain_mem = tm[0,0]
                train_time_pred=joblib.load(conf_dir.joinpath("train_time_RFR_5params.joblib"))
                ##CUDA_cores,MemoryBW_GBps,Memory_clock_speed_MHz,Tensor_cores,SM_count
                #X = df[['activations', 'parameters','input','Pipelines/CUDA cores', 'Memory bandwidth (GB/s)', 'SM count','Memory clock speed (MHz)','Tensor cores']]
                tt = train_time_pred.predict([[float(self.activations),float(self.parameters),float(self.inputsize),gpu[0],gpu[1],gpu[4],gpu[2],gpu[3]]])
                if tt.shape == (1,):
                    train_time = tt[0]
                elif tt.shape == (1,1):
                    train_time = tt[0,0]
                print("%f,%f" % (tain_mem,train_time))

                output['mem'] = tain_mem
                output['time'] = train_time

            ##inference-vgg19-cat2
            if "inference" in self.jobname:
                infer_mem_pred=joblib.load(conf_dir.joinpath("infer_mem_RFR_5params.joblib"))
                #X = df[['activations', 'parameters', 'Memory bandwidth (GB/s)','Pipelines/CUDA cores', 'SM count', 'Memory clock speed (MHz)','Tensor Cores (GPU)']]
                im = infer_mem_pred.predict([[float(self.activations),float(self.parameters),gpu[1],gpu[0],gpu[4],gpu[2],gpu[3]]])
                if im.shape == (1,):
                    infer_mem = im[0]
                elif im.shape == (1,1):
                    infer_mem = im[0,0]

                infer_time_pred=joblib.load(conf_dir.joinpath("infer_time_RFR_5params.joblib"))
                it = infer_time_pred.predict([[float(self.activations),float(self.parameters),gpu[1],gpu[0],gpu[4],gpu[2],gpu[3]]])
                if it.shape == (1,):
                    infer_time = it[0]
                elif it.shape == (1,1):
                    infer_time = it[0,0]
                print("%f,%f" % (infer_mem,infer_time))

                output['mem'] = infer_mem
                output['time'] = infer_time


        elif self.option == "2":
            if "batchsize" in self.jobname:
                train_mem_pred=joblib.load(conf_dir.joinpath("train_mem_RFR_5params.joblib"))
                ##CUDA_cores,MemoryBW_GBps,Memory_clock_speed_MHz,Tensor_cores,SM_count
                #X = df[['activations', 'parameters','input', 'Memory bandwidth (GB/s)', 'Pipelines/CUDA cores','SM count','Memory clock speed (MHz)','Tensor cores']]
                tm = train_mem_pred.predict([[float(self.activations),float(self.parameters),float(self.inputsize),gpu[1],gpu[0],gpu[4],gpu[2],gpu[3]]])
                if tm.shape == (1,):
                    tain_mem = tm[0]
                elif tm.shape == (1,1):
                    tain_mem = tm[0,0]
                train_time_pred=joblib.load(conf_dir.joinpath("train_time_RFR_5params.joblib"))
                ##CUDA_cores,MemoryBW_GBps,Memory_clock_speed_MHz,Tensor_cores,SM_count
                #X = df[['activations', 'parameters','input','Pipelines/CUDA cores', 'Memory bandwidth (GB/s)', 'SM count','Memory clock speed (MHz)','Tensor cores']]
                tt = train_time_pred.predict([[float(self.activations),float(self.parameters),float(self.inputsize),gpu[0],gpu[1],gpu[4],gpu[2],gpu[3]]])
                if tt.shape == (1,):
                    train_time = tt[0]
                elif tt.shape == (1,1):
                    train_time = tt[0,0]
                print("%f,%f" % (tain_mem,train_time))
                output['mem'] = tain_mem
                output['time'] = train_time


            if "inference" in self.jobname:
                infer_mem_pred=joblib.load(conf_dir.joinpath("infer_mem_RFR_3params.joblib"))
                #X = df[['activations', 'parameters', 'Memory bandwidth (GB/s)','Pipelines/CUDA cores', 'SM count']]
                im = infer_mem_pred.predict([[float(self.activations),float(self.parameters),gpu[1],gpu[0],gpu[4]]])
                if im.shape == (1,):
                    infer_mem = im[0]
                elif im.shape == (1,1):
                    infer_mem = im[0,0]

                infer_time_pred=joblib.load(conf_dir.joinpath("infer_time_RFR_3params.joblib"))
                #X = df[['activations', 'parameters', 'Memory bandwidth (GB/s)','Pipelines/CUDA cores', 'SM count']]
                it = infer_time_pred.predict([[float(self.activations),float(self.parameters),gpu[1],gpu[0],gpu[4]]])
                if it.shape == (1,):
                    infer_time = it[0]
                elif it.shape == (1,1):
                    infer_time = it[0,0]
                print("%f,%f" % (infer_mem,infer_time))

                output['mem'] = infer_mem
                output['time'] = infer_time

        if self.json is not None:
            import json
            json_path = Path(self.json)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, 'w') as f:
                json.dump(output, f, indent=4)
        return output

