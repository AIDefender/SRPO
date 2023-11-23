import os
import pickle

import gym
import d4rl
import numpy as np
from loguru import logger

from offlinerl.utils.data import SampleBatch

def load_d4rl_buffer(task):
    env = gym.make(task[5:])
    dataset = d4rl.qlearning_dataset(env)

    buffer = SampleBatch(
        obs=dataset['observations'],
        obs_next=dataset['next_observations'],
        act=dataset['actions'],
        rew=np.expand_dims(np.squeeze(dataset['rewards']), 1),
        done=np.expand_dims(np.squeeze(dataset['terminals']), 1),
    )

    logger.info('obs shape: {}', buffer.obs.shape)
    logger.info('obs_next shape: {}', buffer.obs_next.shape)
    logger.info('act shape: {}', buffer.act.shape)
    logger.info('rew shape: {}', buffer.rew.shape)
    logger.info('done shape: {}', buffer.done.shape)
    logger.info('Episode reward: {}', buffer.rew.sum() /np.sum(buffer.done) )
    logger.info('Number of terminals on: {}', np.sum(buffer.done))
    return buffer

def get_data(dataset, index, cnt):
    if isinstance(dataset, list):
        # dataset_st
        if cnt == 0:
            return [x[index] for x in dataset]
        return [x[index][:cnt] for x in dataset]
    else:
        # dataset_d4rl
        if cnt == 0:
            return dataset[index]
        return dataset[index][:cnt]

def train_test_split(buffer, ratio=0.8):
    # items = []
    keys = ['obs','obs_next','act','rew','done'] 
    # items = [buffer[key] for key in keys]
    dataset_len = buffer[keys[0]].shape[0]
    train_size = int(dataset_len*ratio)
    train_buffer, val_buffer = {},{}
    for key in keys:
        np.random.shuffle(buffer[key])
        # print(buffer[key].shape)
        train_buffer[key] = buffer[key][:train_size]
        val_buffer[key] = buffer[key][train_size:]

    return SampleBatch(**train_buffer), SampleBatch(**val_buffer)

def load_d4rl_st_buffer(task, is_mopo=False, cnt=0, path=None):
    # example task name: body_mass,density_10@walker2d-medium-v0
    env_params, d4rl_env_name = tuple(task.split("@"))
    env = gym.make(d4rl_env_name)
    env_param_lst = env_params.split(",")
    dataset_d4rl = d4rl.qlearning_dataset(env)
    num_origin = len(dataset_d4rl['observations'])
    cnt = int(num_origin * cnt)
    dataset_st = []
    
    for env_param in env_param_lst:
        st_dataset_path = os.path.join(path, 
                                        d4rl_env_name.split("-")[0].capitalize(),
                                        env_param,
                                        d4rl_env_name+".hdf5")
        dataset_st.append(d4rl.qlearning_dataset(env, env.get_dataset(h5path=st_dataset_path)))

    if is_mopo:
        buffer = SampleBatch(
            obs       = np.concatenate((get_data(dataset_d4rl, 'observations', cnt), 
                                      *get_data(dataset_st, 'observations', cnt))),
            obs_next  = np.concatenate((get_data(dataset_d4rl, 'next_observations', cnt), 
                                      *get_data(dataset_st, 'next_observations', cnt))),
            act       = np.concatenate((get_data(dataset_d4rl, 'actions', cnt), 
                                      *get_data(dataset_st, 'actions', cnt))),
            rew       = np.expand_dims(np.concatenate((get_data(dataset_d4rl, 'rewards', cnt), 
                                                       *get_data(dataset_st, 'rewards', cnt))), 1),
            done      = np.expand_dims(np.concatenate((get_data(dataset_d4rl, 'terminals', cnt),
                                                       *get_data(dataset_st, 'terminals', cnt))), 1),
        )
    else:
        buffer = SampleBatch(
            obs       = np.concatenate((get_data(dataset_d4rl, 'observations', cnt), 
                                      *get_data(dataset_st, 'observations', cnt))),
            obs_next  = np.concatenate((get_data(dataset_d4rl, 'next_observations', cnt), 
                                      *get_data(dataset_st, 'next_observations', cnt))),
            act       = np.concatenate((get_data(dataset_d4rl, 'actions', cnt), 
                                      *get_data(dataset_st, 'actions', cnt))),
            rew       = np.expand_dims(np.concatenate((get_data(dataset_d4rl, 'rewards', cnt), 
                                                       *get_data(dataset_st, 'rewards', cnt))), 1),
            done      = np.expand_dims(np.concatenate((get_data(dataset_d4rl, 'terminals', cnt),
                                                       *get_data(dataset_st, 'terminals', cnt))), 1),
            terminals = np.expand_dims(np.concatenate((get_data(dataset_d4rl, 'terminals', cnt), 
                                                       *get_data(dataset_st, 'terminals', cnt))), 1),

            observations      = np.concatenate((get_data(dataset_d4rl, 'observations', cnt), 
                                                *get_data(dataset_st, 'observations', cnt))),
            next_observations = np.concatenate((get_data(dataset_d4rl, 'next_observations', cnt), 
                                                *get_data(dataset_st, 'next_observations', cnt))),
            actions           = np.concatenate((get_data(dataset_d4rl, 'actions', cnt), 
                                                *get_data(dataset_st, 'actions', cnt))),
            rewards           = np.expand_dims(np.concatenate((get_data(dataset_d4rl, 'rewards', cnt), 
                                                               *get_data(dataset_st, 'rewards', cnt))), 1),
        )        
    

    logger.info('target length: {}', dataset_d4rl['observations'].shape[0])
    logger.info('obs shape: {}', buffer.obs.shape)
    logger.info('obs_next shape: {}', buffer.obs_next.shape)
    logger.info('act shape: {}', buffer.act.shape)
    logger.info('rew shape: {}', buffer.rew.shape)
    logger.info('done shape: {}', buffer.done.shape)
    logger.info('Episode reward: {}', buffer.rew.sum() /np.sum(buffer.done) )
    logger.info('Number of terminals on: {}', np.sum(buffer.done))
    return buffer, num_origin