import os
import shutil
import argparse
import random
from loguru import logger


import pandas as pd

import sys
sys.path.insert(0, './baselines/be_great_pafted')
from be_great import GReaT


def prep_data(args):
    # copy data to exp_path
    if not os.path.exists(args.exp_path):
        os.makedirs(args.exp_path)

    shutil.copy(args.data_path, f"{args.exp_path}/real_data.csv")
    data = pd.read_csv(f"{args.exp_path}/real_data.csv")
    return data

def run_training(args, train_data, batch_size=32):
    if args.train:
        model = GReaT(llm='distilgpt2', experiment_dir=f"{args.exp_path}/trainer_great", batch_size=batch_size, epochs=args.epochs)
        model.fit(train_data, having_order=True)

        model_path = f"{args.exp_path}/models/"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model.save(model_path)
    else:
        model = GReaT.load_from_dir(f"{args.exp_path}/models/")
    
    return model

def run_sampling(args, model, n_samples, ext=''):
    # generation
    synthetic_data = model.sample(n_samples=n_samples)
    data_name = os.path.basename(args.data_path).split('.')[0]
    synthetic_data.to_csv(f"{args.exp_path}/synthetic_data{ext}.csv", index=False)
    # copy the synthetic data to data_to_eval folder
    if not os.path.exists(f"./data_to_eval/{args.model}"):
        os.makedirs(f"./data_to_eval/{args.model}")
    shutil.copy(f"{args.exp_path}/synthetic_data{ext}.csv", f"./data_to_eval/{args.model}/{args.exp_path.split('/')[-1]}{ext}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Runner")
    parser.add_argument("--model", default=f"paft", help="model name")
    parser.add_argument("--data_path", default=f"./data/bird_geolocation", help="data file")
    parser.add_argument("--train", action="store_true", help="Enable train mode")
    parser.add_argument("--order", default=None, type=str, help="Wanted order")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs")
    parser.add_argument("--exptimes", default=5, type=int, help="Number of experiments")
    
    args = parser.parse_args()
    
    print(args)
    log_folder = 'run_logs_neurips'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    logger.add(f"{log_folder}/loguru_{args.model}.log")

    task_list = [
        './data/train/travel.csv',
        './data/train/bejing.csv',
        './data/train/us_location.csv',
        './data/train/california_housing.csv',
        './data/train/adult.csv',
        './data/train/seattle_housing.csv',
    ]
    with open('feature_order_permutation.txt', 'r') as f:
        sorted_order_dict = eval(f.read())

    for task in task_list:
        args.data_path = task

        todo_list = []
        if os.path.isfile(args.data_path):
            todo_list.append(args.data_path)
        else:
            for file in os.listdir(args.data_path):
                if file.endswith(".csv"):
                    todo_list.append(os.path.join(args.data_path, file))
        
        logger.info(f"todo: {todo_list}")

        for data_path in todo_list:
            args.data_path = data_path
            args.exp_path = f"./run_output/{args.model}/{os.path.basename(args.data_path).split('.')[0]}"
            logger.info(f"running {args.data_path}")
            
            tmp_path = args.exp_path
            
            time_begin = pd.Timestamp.now()
            args.exp_path = f"{tmp_path}"
            if not os.path.exists(args.exp_path):
                os.makedirs(args.exp_path)
        
            data = prep_data(args)
            if args.model == 'paft':
                args.order = sorted_order_dict[os.path.basename(args.data_path).split('.')[0]]
                if args.order == '':
                    # no FD in the dataset
                    args.order = data.columns.tolist() # a fixed random order for all samples
                    # permute the order
                    random.shuffle(args.order)
                    args.order = ','.join(args.order)
                logger.info(f"reordering to {args.order}")
                print(f"reordering to {args.order}")
                data = data[args.order.split(',')]

            # drop rows with NaN
            data = data.dropna()
            batch_size = 32
            model = run_training(args, data, batch_size=batch_size)
            time_train_end = pd.Timestamp.now()
            for i in range(args.exptimes):
                logger.info(f"running {i}th experiment")
                time_sample_begin = pd.Timestamp.now()
                run_sampling(args, model, len(data), ext=f"_{i}")
                time_end = pd.Timestamp.now()
                
                with open(f"{log_folder}/time_paft.txt", "a") as f:
                    f.write(f"paft,{os.path.basename(args.data_path)},{i},{(time_train_end - time_begin).seconds}, {(time_end - time_sample_begin).seconds}, {args.order}\n")
                    