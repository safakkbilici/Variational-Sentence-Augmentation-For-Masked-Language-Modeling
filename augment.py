from models.variational_gru import VariationalGRU
from utils.model_utils import to_var, idx2word, interpolate
import json
import os
import torch
from tqdm import tqdm

if __name__ == "__main__":
    parser.add_argument('--data_name', type=str, default='data')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('-bin', '--save_model_path', type=str, default='models')
    parser.add_argument('-ckpt', '--checkpoint', type=str)

    with open(args.data_dir+f'/{args.data_name}.vocab.json', 'r') as f:
        vocab = json.load(f)

    with open(os.path.join(args.model_path, "model_params.json"), "r") as f:
        params = json.load(f)

    load_checkpoint = args.checkpoint

    model = VariationalGRU(**params)
    model.load_state_dict(torch.load(load_checkpoint))

    if params["device"] == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    model = model.to(device)







