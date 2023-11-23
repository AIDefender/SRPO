import fire

from offlinerl.algo import algo_select
from offlinerl.data.d4rl import load_d4rl_buffer, load_d4rl_st_buffer, train_test_split
from offlinerl.evaluation import OnlineCallBackFunction, OnlineMultiEnvCallBackFunction
from torch import set_num_threads

def run_algo(**kwargs):
    algo_init_fn, algo_trainer_obj, algo_config = algo_select(kwargs)
    set_num_threads(algo_config['num_threads'])
    if "@" in algo_config["task"]:
        train_buffer, num_origin = load_d4rl_st_buffer(algo_config["task"], 
                                                       'mopo' in algo_config['algo_name'] or 'bc' in algo_config['algo_name'],
                                                       algo_config['cnt'],
                                                       path="/data/home/zhxue/offline/state_prior/NeurIPS2023_Dataset")
        algo_config["task_full"] = algo_config["task"]
        algo_config["task"] = "d4rl-" + algo_config["task"].split("@")[1]
    else:
        train_buffer = load_d4rl_buffer(algo_config["task"])
    
    # determine the number of source and target samples for distinguishing
    # between the two
    algo_config['num_origin'] = num_origin
    algo_init = algo_init_fn(algo_config)
    algo_trainer = algo_trainer_obj(algo_init, algo_config)
    callback = OnlineMultiEnvCallBackFunction()
    callback.initialize(train_buffer=train_buffer, val_buffer=None, task=algo_config["task_full"])

    if 'bc' in algo_config['algo_name']:
        train_buffer, val_buffer = train_test_split(train_buffer)
    else:
        val_buffer = None
    algo_trainer.train(train_buffer, val_buffer, callback_fn=callback)

if __name__ == "__main__":
    fire.Fire(run_algo)
    