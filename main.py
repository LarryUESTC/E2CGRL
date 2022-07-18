import numpy as np
import random as random
import torch
from params import parse_args,printConfig
import os
from models.E2_SGRL import E2_SGRL


def main(dataset:str):
    # param set
    args, unknown = parse_args(dataset)
    printConfig(args)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    embedder = E2_SGRL(args)

    # output setting
    filePath = "log"
    exp_ID = 0
    for filename in os.listdir(filePath):
        file_info = filename.split("_")
        file_dataname = file_info[0]
        if file_dataname == args.dataset:
            exp_ID = max(int(file_info[1].split('.')[0]), exp_ID)
    exp_name = args.dataset + "_" + str(exp_ID + 1)+'.txt'
    exp_name = os.path.join(filePath, exp_name)
    arg_file = open(exp_name,"a")
    for k, v in sorted(args.__dict__.items()):
        arg_file.write("\n- {}: {}".format(k, v))

    ## train
    macro_f1s, micro_f1s, k1, k2, st,training_time = embedder.training()
    arg_file.write(
        "\n- macro_f1s:{},std:{}, micro_f1s:{},std:{},k1:{},std:{},training time:{}s".format(np.mean(macro_f1s), np.std(macro_f1s),
                                                                           np.mean(micro_f1s), np.std(micro_f1s),
                                                                           np.mean(k1), np.std(k1),
                                                                                             training_time))
    arg_file.close()


if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dataset = 'dblp' # choice: acm dblp imdb freebase
    main(dataset)
