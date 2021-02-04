#%%
import torch
import pdb
from tqdm import tqdm
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers import FastaiLRFinder
from monai.data import DataLoader
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import Checkpoint,DiskSaver,TerminateOnNan,EarlyStopping
import mlflow
from ignite.engine import _prepare_batch,create_supervised_evaluator,Events,create_supervised_trainer
#%%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#%%
class ClassificationTask():
    def __init__(self):
        super().__init__()
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.maxepochs = 3
        self.batchsize = 2
        self.dataset = {}
        self.lr_find = True
        self.resume = False
        self.early_stopping = True
        self.dev = device
        self.checkpt_path = "checkpt/trainer/checkpoint_34.pt"
        self.state_path = "checkpt/training_states/dummytest.pt" 
    def adding_Event_handlers(self,trainer):
        log_interval = 1
        desc = "Mini-batch loss = {:.2f}"
        pbar = tqdm(initial=0, leave=False, total=len(self.train_loader),desc=desc.format(0))
        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
        def output_transform(output):
            y_pred = output["y_pred"]
            y = output["y"]
            return (y_pred,y)
        acc_metric = Accuracy(output_transform)
        ce_metric = Loss(self.criterion,output_transform = output_transform)
        acc_metric.attach(trainer,"Accuracy")
        ce_metric.attach(trainer,"loss")
        @trainer.on(Events.ITERATION_COMPLETED(every = log_interval))
        def display_minibatch_loss(engine):
            pbar.desc = desc.format(engine.state.output["loss"])
            pbar.update(log_interval)
        if self.early_stopping:
            def score_func(engine):
                score = self.criterion(engine.state.output[0],engine.state.output[1])
                return -score
            es_handler = EarlyStopping(patience = 10,min_delta = 0.0,score_function = score_func, trainer = trainer)
            self.evaluator.add_event_handler(Events.COMPLETED, es_handler)
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            pbar.refresh()
            self.train_evaluator.run(self.train_loader)
            metrics = self.train_evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_nll = metrics['nll']
            tqdm.write("Training Results - Epoch: {}/{}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(engine.state.epoch,engine.state.max_epochs, avg_accuracy, avg_nll))
            mlflow.log_metric("train_accuracy",avg_accuracy)        
            mlflow.log_metric("train_loss",avg_nll)
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            self.evaluator.run(self.valid_loader)
            metrics = self.evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_nll = metrics['nll']
            tqdm.write("Validation Results - Epoch: {}/{}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(engine.state.epoch,engine.state.max_epochs, avg_accuracy, avg_nll))
            print("updated_lr = ",self.optimizer.param_groups[0]['lr'])
            mlflow.log_metric("lr",self.optimizer.param_groups[0]['lr'])
            mlflow.log_metric("valid_accuracy",avg_accuracy)
            mlflow.log_metric("valid_loss",avg_nll)
            pbar.n = pbar.last_print_n = 0
        @self.evaluator.on(Events.COMPLETED)
        def update_lr(engine):
            self.scheduler.step(engine.state.metrics["nll"])
        @trainer.on(Events.COMPLETED)
        def exiting_run(engine):
            print('\033[95m' + '\t\t\t\t\t\t\t\t---Training Completed!?---' + '\033[0m')
            pbar.close()
    def get_trainer(self):
        self.model = self.model.to(self.dev)
        self.train_loader = DataLoader(self.dataset["train"], batch_size = self.batchsize,shuffle = True)
        self.valid_loader = DataLoader(self.dataset["valid"], batch_size = self.batchsize,shuffle = True)
        def prep_batch(batch,device = device, non_blocking = False):
            return _prepare_batch((batch["img"],torch.squeeze(batch["label"],1).type(torch.LongTensor)), device, non_blocking)
        def trainer_output_transform(x,y,y_pred,loss):
            return {"input":x,"y":y,"y_pred":y_pred,"loss":loss.item()}
        trainer = create_supervised_trainer(
                                            device = self.dev,
                                            model = self.model,
                                            optimizer = self.optimizer,
                                            loss_fn = self.criterion,
                                            prepare_batch = prep_batch,
                                            output_transform = trainer_output_transform
                                            )
        trainer.load_state_dict({"epoch":0,"epoch_length":len(self.train_loader),"max_epochs":self.maxepochs})
        if self.resume:
            print("\n Loading the saved model(resuming)\n")
            to_load = {'trainer': trainer, 'model': self.model, 'optimizer': self.optimizer,"lr_scheduler":self.scheduler}
            checkpoint = torch.load(self.checkpt_path)
            Checkpoint.load_objects(to_load = to_load, checkpoint = checkpoint)
            end_epoch = trainer.state.max_epochs
            trainer.load_state_dict({"epoch":end_epoch,"epoch_length":len(self.train_loader),"max_epochs":4})
        if self.lr_find:
            dummy_trainer = create_supervised_trainer(
                                                      device = self.dev,
                                                      model = self.model,
                                                      optimizer = self.optimizer,
                                                      loss_fn = self.criterion,
                                                      prepare_batch = prep_batch,
                                                     )
            print("\nOld lr =",self.optimizer.param_groups[0]['lr'],"\n")
            lr_finder = FastaiLRFinder()
            to_save={'model': self.model, 'optimizer': self.optimizer}
            with lr_finder.attach(dummy_trainer,to_save,end_lr = 0.1) as trainer_with_lr_finder:
                trainer_with_lr_finder.run(data = self.train_loader,max_epochs = 7)
            del dummy_trainer
            lr_finder.plot()
            self.optimizer.param_groups[0]['lr'] = lr_finder.lr_suggestion()
            print("\nNew lr suggested by lr_finder =",lr_finder.lr_suggestion(),"\n")
        self.evaluator = create_supervised_evaluator(self.model,metrics={'accuracy': Accuracy(),'nll': Loss(self.criterion)},prepare_batch = prep_batch,device = self.dev)
        self.train_evaluator = create_supervised_evaluator(self.model,metrics={'accuracy': Accuracy(),'nll': Loss(self.criterion)},prepare_batch = prep_batch,device = self.dev)        
        if False:
            torch.save(trainer.state_dict(), self.state_path)
            to_save = {'trainer': trainer, 'model': self.model, 'optimizer': self.optimizer,"lr_scheduler":self.scheduler}
            checkpt_handler = Checkpoint(to_save, DiskSaver("checkpt/trainer/", create_dir = True,require_empty = False))
            trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpt_handler) # Is the checkpointing done at the last-epoch or at every epoch?
        return trainer