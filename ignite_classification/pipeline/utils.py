import inspect
import json
import argparse
import sys
import pdb
from pipeline.task import *
from pipeline import utils_torch
from pydoc import locate
# from . import utils_torch

#%%
# pass args named with a.b, get a nested dict
def unflatten(dictobj):

    def split_rec(k,v,out):        
        if '.' in k:
            pre,rest = k.split('.',1)
            split_rec(rest,v,out.setdefault(pre,{}))
        else:
            out[k]=v

    res = {}
    for k,v in dictobj.items():
        split_rec(k,v,res)        
        
    return res

#%%
def check_update(dictobj1, dictobj2):    
    # print(dictobj1)
    # print(dictobj2)
    for key in dictobj2:
        if type(dictobj2[key])==dict:
            # print('dictkey %s' % key)
            if key in dictobj1:
                check_update(dictobj1[key], dictobj2[key])            
            else:    
                dictobj1[key]={}
                dictobj1[key].update(dictobj2[key])
        else:
            # print('key %s' % key)
            if dictobj2[key] is not None:
                dictobj1[key]=dictobj2[key]
        
#%%
# vars(args) will mostly do

# def props(obj):
#     # Argparse to json
#     pr = {}
#     for name in dir(obj):
#         value = getattr(obj, name)
#         if not name.startswith('__') and not inspect.ismethod(value):
#             pr[name] = value
#     return pr

#%%
class ArgParser:
    """Takes a flat json file specifying argument names - eg argnames.json, 
        and creates argparse object. Provides parse_args, returning command line args as dict
    """
    def __init__(self,jsonfile,title=None):
        self.parser = argparse.ArgumentParser(description=title,
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('cfgname')
        for key, val in json.load(open(jsonfile)).items():
            ar = ['--%s' % key]
            kwar = val
            if 'default' in val:                
                kwar['type']=type(val['default'])
            if 'type' in val:
                kwar['type']=locate(val['type'])

            self.parser.add_argument(*ar,**kwar)

    def parse_args(self,argsin):
        # returns as flat dict
        return vars(self.parser.parse_args(argsin))

#%%
def instantiate(typename, args=None, kwargs=None):
    print('instantiate: got type %s' % typename)    
    obj = locate(typename)
    if obj is None:
        raise Exception("instantiate: typename not known: %s" % typename)
    if args is None and kwargs is None:
        return obj()
    if args is not None:
        print('\targs:',args)
        if kwargs is None:
            if type(args)==list:
                inst = obj(*args)
            else:
                inst = obj(args)
        else:
            print('\tkwargs:',kwargs)
            inst = obj(*args,**kwargs)        
    else:
        print('\tkwargs:',kwargs)
        inst = obj(**kwargs)
    return inst

#%%
class JsonConfig:

    def __init__(self,filename):
        self.config = json.load(open(filename))

    def __repr__(self):
        return str(self.config)

    def update_from_args(self,args):        
        check_update(self.config, unflatten(args))

    def get_datahelper_instance(self):
        typename = self.config['DATASET']['DH_TYPE']
        
        args = {}
        if 'PARAMS' in self.config['DATASET']:
            args = self.config['DATASET']['PARAMS']

        return instantiate(typename, args)

    def get_model_instance(self):
        typename = self.config['MODEL']['MODEL_TYPE']
        
        kwargs = None
        if 'MODEL_ARGS' in self.config['MODEL']:
            kwargs = self.config['MODEL']['MODEL_ARGS']
        
        return instantiate(typename, None, kwargs)

    def get_criterion(self):
        _conf = self.config['OPTIM']
        cname = _conf['Criterion']
        
        typename = utils_torch.loss_by_name[cname]

        kwargs = None
        if 'loss_args' in _conf:
            kwargs = _conf['loss_args']

        return instantiate(typename, None, kwargs) 

    def get_optimizer(self,modelparams):
        _conf = self.config['OPTIM']
        methodname = _conf['Method']
        
        typename = utils_torch.optimizer_by_name[methodname]

        kwargs = None
        if 'optim_args' in _conf:
            kwargs = _conf['optim_args']

        optimizer = instantiate(typename, [modelparams], kwargs)

        scheduler = None
        if 'lr_schedule' in _conf:
            if 'active' in _conf['lr_schedule']:
                if _conf['lr_schedule']['active']:
                    typename = utils_torch.scheduler_by_name[_conf['lr_schedule']['type']]
                    kwargs = _conf['lr_schedule']['options']
                    kwargs["optimizer"] = optimizer
                    scheduler = instantiate(typename, None, kwargs)
                    
        return optimizer, scheduler

    def get_training_params(self):
        # FIXME: enforced args to be checked, defaults to be set
        return self.config['TRAINING']
    
    def get_task(self):
        if self.config['TASK'] == "classification":
            task = ClassificationTask()
            if "Batch_size" in self.config["TRAINING"]:
                task.batchsize = self.config["TRAINING"]["Batch_size"]
            if "Num_epochs" in self.config["TRAINING"]:
                task.maxepochs = self.config["TRAINING"]["Num_epochs"]
            if "resume" in self.config["TRAINING"]:
                task.resume = self.config["TRAINING"]["resume"]
                if self.config["TRAINING"]["resume"]:
                    task.checkpt_path = "checkpt/trainer/checkpoint_68.pt"
            return task
        return None

#%%
# def parseInputs(argsin=sys.argv,title=None):
def parseInputs(argsin=["train.py","classification/confs/brats.json"],title=None):
    """for command line parsing
        first argument is always cfg json filename. 
        overrides follow with --<group>.<param>
    """
    my_args = None
    cfg = None
    if len(argsin)>2:
        my_args = ArgParser('argnames.json',title).parse_args(argsin[1:])
        
        if 'cfgname' in my_args:
            cfg = JsonConfig(my_args['cfgname'])
            del(my_args['cfgname'])
    else:
        cfg = JsonConfig(argsin[1])    
    return cfg, my_args