from models.variational_gru import VariationalGRU
from utils.model_utils import to_var, idx2word, interpolate
from utils.data_utils import clean_sentences, select_k, save_sentences, clear_duplicates

import json, os, torch, argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('-dn','--data_name', type=str, default='data')
    parser.add_argument('-ddir','--data_dir', type=str, default='data')
    parser.add_argument('-bin', '--save_model_path', type=str, default='models')
    parser.add_argument('-ckpt', '--checkpoint', type=str)
    parser.add_argument('-g', '--generate_iteration', type=int, default=10_000)
    parser.add_argument('-n', '--num_samples', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=10)
    parser.add_argument('-ml', '--min_sentence_length', type=int, default=3)
    parser.add_argument('-ut', '--unk_threshold', type=int, default=3)
    args = parser.parse_args()

    with open(args.data_dir+f'/{args.data_name}.vocab.json', 'r') as f:
        vocab = json.load(f)

    w2i, i2w = vocab['w2i'], vocab['i2w']
    with open(os.path.join(args.save_model_path, "model_params.json"), "r") as f:
        params = json.load(f)

    load_checkpoint = args.checkpoint

    model = VariationalGRU(**params)
    model.load_state_dict(torch.load(load_checkpoint))

    if params["device"] == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    latent_size = params["latent_size"]
    population_sentences = []
    for randomness in tqdm(range(args.generate_iteration)):
        model.eval()
        samples, z = model.inference(n=args.num_samples)
        samples = idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>'])

        z1 = torch.randn([latent_size]).numpy()
        z2 = torch.randn([latent_size]).numpy()
        z3 = to_var(torch.from_numpy(interpolate(start=z1, end=z2, steps=3)).float())
        samples2, _ = model.inference(z=z3)
        samples2 = idx2word(samples2, i2w=i2w, pad_idx=w2i['<pad>'])

        sentences1 = select_k(samples, unk_threshold=args.unk_threshold, k = args.min_sentence_length)
        sentences2 = select_k(samples2, unk_threshold=args.unk_threshold, k = args.min_sentence_length)

        if sentences1:
            population_sentences.append(sentences1)
        if sentences2:
            population_sentences.append(sentences2)


    save_sentences(population_sentences, file = 'generated_all.txt', population=True)

    cleaned = clean_sentences('generated_all.txt')
    save_sentences(cleaned, file = 'cleaned.txt')

    clear_duplicates('cleaned.txt')
