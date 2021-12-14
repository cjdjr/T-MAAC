import numpy as np
import os
import argparse
import yaml
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.var_voltage_control.voltage_control_env import VoltageControl


parser = argparse.ArgumentParser(description="Train rl agent.")
parser.add_argument("--save-path", type=str, nargs="?", default="./", help="Please enter the directory of saving model.")
parser.add_argument("--length", type=int, nargs="?", default=2)
parser.add_argument("--env", type=str, nargs="?", default="var_voltage_control", help="Please enter the env name.")
parser.add_argument("--scenario", type=str, nargs="?", default="case33_3min_final", help="Please input the valid name of an environment scenario.")
parser.add_argument("--season", type=str, nargs="?", default="all", help="all/summer/winter")
parser.add_argument("--date-emb",  action='store_true')
parser.add_argument("--mode", type=str, nargs="?", default="centralised", help="Please enter the mode: distributed or decentralised.")
argv = parser.parse_args()

# load env args
with open("./args/env_args/"+argv.env+".yaml", "r") as f:
    env_config_dict = yaml.safe_load(f)["env_args"]
data_path = env_config_dict["data_path"].split("/")
data_path[-1] = argv.scenario
env_config_dict["data_path"] = "/".join(data_path)
net_topology = argv.scenario

# set the action range
assert net_topology in ['case33_3min_final', 'case141_3min_final', 'case322_3min_final'], f'{net_topology} is not a valid scenario.'
if argv.scenario == 'case33_3min_final':
    env_config_dict["action_bias"] = 0.0
    env_config_dict["action_scale"] = 0.8
elif argv.scenario == 'case141_3min_final':
    env_config_dict["action_bias"] = 0.0
    env_config_dict["action_scale"] = 0.6
elif argv.scenario == 'case322_3min_final':
    env_config_dict["action_bias"] = 0.0
    env_config_dict["action_scale"] = 0.8

assert argv.mode in ['distributed', 'decentralised', 'centralised'], "Please input the correct mode, e.g. distributed or  or centralised."
env_config_dict["mode"] = argv.mode

assert argv.season in ['all', 'summer', 'winter'], "Please input the correct season, e.g. all or summer or winter."
env_config_dict["season"] = argv.season

if argv.date_emb:
    env_config_dict["state_space"].append("date")

# load default args
with open("./args/default.yaml", "r") as f:
    default_config_dict = yaml.safe_load(f)


# define envs
env = VoltageControl(env_config_dict)
bus_num = len(env.base_powergrid.bus)
cnt = 0
data = {
    'state': [],
    'q': [],
    'res_v': [],
}
while cnt < argv.length:
    _, state = env.reset()
    done = False
    while not done:
        actions = env.get_action() # random_action
        reward, done, info = env.step(actions)
        next_state = env.get_state()

        data['state'].append(state)
        data['q'].append(env.now_q)
        data['res_v'].append(next_state[-2*bus_num:-bus_num])
        cnt+=1
        
        if cnt % 1000 == 0:
            print("nums : {} ".format(cnt))

        state = next_state

data['state'] = np.array(data['state'])
data['q'] = np.array(data['q'])
data['res_v'] = np.array(data['res_v'])

np.save('transition/{}.npy'.format(argv.scenario),data)