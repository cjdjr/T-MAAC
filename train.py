import torch as th
import os
import argparse
import yaml
from tensorboardX import SummaryWriter

from models.model_registry import Model, Strategy
from environments.var_voltage_control.voltage_control_env import VoltageControl
from utilities.util import convert, dict2str
from utilities.trainer import PGTrainer
import wandb
from transition.model import transition_model, transition_model_linear

parser = argparse.ArgumentParser(description="Train rl agent.")
parser.add_argument("--save-path", type=str, nargs="?", default="./",
                    help="Please enter the directory of saving model.")
parser.add_argument("--alg", type=str, nargs="?",
                    default="maddpg", help="Please enter the alg name.")
parser.add_argument("--env", type=str, nargs="?",
                    default="var_voltage_control", help="Please enter the env name.")
parser.add_argument("--alias", type=str, nargs="?", default="",
                    help="Please enter the alias for exp control.")
parser.add_argument("--mode", type=str, nargs="?", default="distributed",
                    help="Please enter the mode: distributed or decentralised.")
parser.add_argument("--scenario", type=str, nargs="?", default="case33_3min_final",
                    help="Please input the valid name of an environment scenario.")
parser.add_argument("--qweight", type=float, nargs="?", default=0.1,
                    help="Please input the q weight of env: 0.01 for case141 and 0.1 for case322")
parser.add_argument("--voltage-barrier-type", type=str, nargs="?", default="l1",
                    help="Please input the valid voltage barrier type: l1, courant_beltrami, l2, bowl or bump.")
parser.add_argument("--season", type=str, nargs="?",
                    default="all", help="all/summer/winter")
# parser.add_argument("--date-emb",  action='store_true')
# parser.add_argument("--safe-trans",  action='store_true')
parser.add_argument("--wandb",  action='store_true')
# parser.add_argument("--safe", type=str, nargs="?",
#                     default="none", help="none/hard/soft")
# parser.add_argument("--constraint_model_path", type=str,
#                     nargs="?", default="transition/case322_3min_final.lin_model")
argv = parser.parse_args()

# load env args
with open("./args/env_args/"+argv.env+".yaml", "r") as f:
    env_config_dict = yaml.safe_load(f)["env_args"]
data_path = env_config_dict["data_path"].split("/")
data_path[-1] = argv.scenario
env_config_dict["data_path"] = "/".join(data_path)
net_topology = argv.scenario

# set the action range
assert net_topology in ['case33_3min_final', 'case141_3min_final',
                        'case322_3min_final'], f'{net_topology} is not a valid scenario.'
if argv.scenario == 'case33_3min_final':
    env_config_dict["action_bias"] = 0.0
    env_config_dict["action_scale"] = 0.8
elif argv.scenario == 'case141_3min_final':
    env_config_dict["action_bias"] = 0.0
    env_config_dict["action_scale"] = 0.6
elif argv.scenario == 'case322_3min_final':
    env_config_dict["action_bias"] = 0.0
    env_config_dict["action_scale"] = 0.8

assert argv.mode in ['distributed', 'decentralised',
                     'centralised'], "Please input the correct mode, e.g. distributed or  or centralised."
env_config_dict["mode"] = argv.mode
env_config_dict["voltage_barrier_type"] = argv.voltage_barrier_type

assert argv.season in [
    'all', 'summer', 'winter'], "Please input the correct season, e.g. all or summer or winter."
env_config_dict["season"] = argv.season

env_config_dict["q_weight"] = argv.qweight

# if argv.date_emb:
#     env_config_dict["state_space"].append("date")

# load default args
with open("./args/default.yaml", "r") as f:
    default_config_dict = yaml.safe_load(f)

# define envs
if argv.env == "var_voltage_control":
    env = VoltageControl(env_config_dict)
    default_config_dict["continuous"] = True

# load alg args
with open("./args/alg_args/" + argv.alg + ".yaml", "r") as f:
    alg_config_dict = yaml.safe_load(f)["alg_args"]
    alg_config_dict["action_scale"] = env_config_dict["action_scale"]
    alg_config_dict["action_bias"] = env_config_dict["action_bias"]

log_name = "-".join([argv.env, net_topology, argv.mode,
                    argv.alg, argv.voltage_barrier_type, argv.alias])
alg_config_dict = {**default_config_dict, **alg_config_dict}

alg_config_dict["agent_num"] = env.get_num_of_agents()
alg_config_dict["obs_size"] = env.get_obs_size()
alg_config_dict["obs_bus_dim"] = env.get_obs_dim()
alg_config_dict["obs_bus_num"] = env.get_obs_bus_num()
alg_config_dict["action_dim"] = env.get_total_actions()
alg_config_dict["bus_num"] = env.get_num_of_buses()
# alg_config_dict["obs_position_list"] = env.get_obs_position_list()
alg_config_dict["region_num"] = env.get_num_of_regions()
alg_config_dict['constraint_mask'] = env.get_constraint_mask()
alg_config_dict['agent2region'] = env.get_agent2region()
alg_config_dict['agent_index_in_obs'] = env.get_agent_index_in_obs()
alg_config_dict['region_adj'] = env.get_region_adj()

# if argv.date_emb:
#     alg_config_dict['agent_type'] = "rnn_with_date"
#     alg_config_dict['use_date'] = True

# if argv.safe_trans:
#     alg_config_dict['safe_trans'] = True

constraint_model = None
# alg_config_dict['safe_filter'] = argv.safe
# if argv.safe != 'none':
#     alg_config_dict['constraint_model_path'] = argv.constraint_model_path
#     device = th.device("cuda" if th.cuda.is_available()
#                        and alg_config_dict['cuda'] else "cpu")
#     constraint_model = transition_model_linear().to(device)
#     constraint_model.load_state_dict(th.load(argv.constraint_model_path))

# If you want to use wandb, please replace project and entity with yours.
if argv.wandb:
    wandb.init(
        project='mapdn_model-based',
        entity="chelly",
        name=log_name,
        group='_'.join(log_name.split('_')[:-1]),
        save_code=True
    )
    wandb.config.update(env_config_dict)
    wandb.config.update(alg_config_dict)
    wandb.run.log_code('.')


args = convert(alg_config_dict)

# define the save path
if argv.save_path[-1] is "/":
    save_path = argv.save_path
else:
    save_path = argv.save_path+"/"

# create the save folders
if "model_save" not in os.listdir(save_path):
    os.mkdir(save_path + "model_save")
if "tensorboard" not in os.listdir(save_path):
    os.mkdir(save_path + "tensorboard")
if log_name not in os.listdir(save_path + "model_save/"):
    os.mkdir(save_path + "model_save/" + log_name)
if log_name not in os.listdir(save_path + "tensorboard/"):
    os.mkdir(save_path + "tensorboard/" + log_name)
else:
    path = save_path + "tensorboard/" + log_name
    for f in os.listdir(path):
        file_path = os.path.join(path, f)
        if os.path.isfile(file_path):
            os.remove(file_path)

# create the logger
logger = SummaryWriter(save_path + "tensorboard/" + log_name)

model = Model[argv.alg]

strategy = Strategy[argv.alg]

print(f"{args}\n")

if strategy == "pg":
    train = PGTrainer(args, model, env, logger, constraint_model)
elif strategy == "q":
    raise NotImplementedError("This needs to be implemented.")
else:
    raise RuntimeError("Please input the correct strategy, e.g. pg or q.")

with open(save_path + "tensorboard/" + log_name + "/log.txt", "w+") as file:
    alg_args2str = dict2str(alg_config_dict, 'alg_params')
    env_args2str = dict2str(env_config_dict, 'env_params')
    file.write(alg_args2str + "\n")
    file.write(env_args2str + "\n")

for i in range(args.train_episodes_num):
    stat = {}
    train.run(stat, i)
    train.logging(stat, argv.wandb)
    if i % args.save_model_freq == args.save_model_freq-1:
        train.print_info(stat)
        th.save({"model_state_dict": train.behaviour_net.state_dict()},
                save_path + "model_save/" + log_name + "/model.pt")
        print("The model is saved!\n")

logger.close()
