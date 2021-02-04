#%%
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
#%%
# %reset -f
import torch
import mlflow
import mlflow.pytorch
from torchsummary import summary
from mlflow.tracking import MlflowClient
from pipeline.utils import parseInputs
#%%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
#%%
def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))
#%%
def train(cfg, overrides=None):
    with mlflow.start_run() as run:
        if overrides is not None:
            cfg.update_from_args(overrides)
        
        # mlflow.log_params(cfg.config)
        
        task = cfg.get_task()
        
        task.model = cfg.get_model_instance()
        
        task.dataset = cfg.get_datahelper_instance()
        
        task.criterion = cfg.get_criterion()
        
        task.optimizer, task.scheduler = cfg.get_optimizer(task.model.parameters())
        
        # summary(task.model,(1,128,128,128))
        
        change_max_epochs = False
        if change_max_epochs:
            task.maxepochs = 3
        
        change_batchsize = False
        if change_batchsize:
            task.batchsize = 1
        
        trainer = task.get_trainer()
        
        trainer.run()
        
        # mlflow.pytorch.log_model(task.model, "model")
        # print(run.info, run.data)
        print_auto_logged_info(mlflow.get_run(run_id = run.info.run_id))
#%%
cfg, my_args = parseInputs()
#%%
train(cfg = cfg,overrides = my_args)
# %%
