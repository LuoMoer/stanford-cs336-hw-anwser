from utils import *
from modeling import *
import argparse
import numpy as np
import torch
import json
from tqdm import tqdm
import swanlab

def get_args():
    parser = argparse.ArgumentParser(description="train_info")
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--valid_data_path", type=str)

    args = parser.parse_args()
    return args

def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function,
    train_data, valid_data,
    max_steps: int,
    batch_size: int,
    max_seq_len: int,
    log_step: int,
    valid_step: int,
    valid_batch_size: int,
    save_step: int,
    save_path: str,
    start_t = 0,
    train_name = "train_test",
    clip_grad_norm = 1.0
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    t_loop = range(start_t, max_steps, 1)
    SAVE_DIR = os.path.join(save_path, train_name)
    
    for t in tqdm(t_loop):
        model.train()
        optimizer.zero_grad()
        x, labels = data_loading(train_data, batch_size, max_seq_len, device)
        y = model(x)
        loss = loss_function(
            y.reshape(-1, y.shape[-1]),
            labels.reshape(-1)
        )
        loss.backward()
        gradient_clipping(model.parameters(), clip_grad_norm)
        optimizer.step()
        
        if t % log_step == 0:
            # print(f"Step [{t}]: train loss = {loss}")
            swanlab.log({"train_loss": loss} , step=t)
        
        if t % valid_step == 0:
            model.eval()
            with torch.no_grad():
                data_iter = valid_data_loading(valid_data, valid_batch_size, max_seq_len, device)
                loss = 0.0
                item_sum = 0
                valid_t = len(valid_data+max_seq_len*valid_batch_size-1)//(max_seq_len*valid_batch_size)
                for x, labels in data_iter:
                    y = model(x)

                    # TODO not end
                    loss += loss_function(
                        y.reshape(-1, y.shape[-1]),
                        labels.reshape(-1)
                    ).item()*len(y)
                    item_sum += len(y)
                    torch.cuda.empty_cache()
                valid_loss = loss/item_sum
                # print(f"Step [{t}]: valid loss = {valid_loss}")
                swanlab.log({"valid_loss": valid_loss}, step=t)
                
        
        if t % save_step == 0:
            if not os.path.exists(SAVE_DIR):
                os.makedirs(SAVE_DIR)
            SAVE_PATH = os.path.join(SAVE_DIR, f"checkpoint-{t}.pth")
            save_checkpoint(model, optimizer, t, SAVE_PATH)

def main():
    args = get_args()
    CONFIG_PATH = args.config_path
    TRAIN_DATA_PATH = args.train_data_path
    VALID_DATA_PATH = args.valid_data_path

    train_data = np.load(TRAIN_DATA_PATH, mmap_mode="r")
    valid_data = np.load(VALID_DATA_PATH, mmap_mode="r")

    with open(CONFIG_PATH, "r") as f:
        configs = json.load(f)
    
    for config in configs:
        model_config = config["model_config"]
        optimizer_config = config["optimizer_config"]
        train_config = config["train_config"]
        
        model = transformer_lm(**model_config)
        optimizer = AdamW(
            model.parameters(), learning_rate_schedule, **optimizer_config)
        loss_function = cross_entropy
        
        train_name = config["train_name"]
        swanlab.init(
            project = "CS336",
            name = train_name,
            config = train_config,
        )
        train(model, optimizer, loss_function, train_data, valid_data, **train_config, train_name=train_name)
        swanlab.finish()
        # rerun train
        # CHECKPOINT_PATH = "/home/swluo/data/CS336/assignment1-basics/cs336_basics/checkpoint/train_test/checkpoint-20.pth"
        # start_t = load_checkpoint(CHECKPOINT_PATH, model, optimizer)
        # train(model, optimizer, loss_function, train_data, valid_data, **train_config, start_t=start_t)

if __name__ == "__main__":
    main()