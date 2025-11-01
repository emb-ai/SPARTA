from args_utils import parse_args
from arca import run_arca
from model_utils import get_raw_embedding_table, get_model_and_tokenizer
from utils import load_outputs
from tqdm import tqdm

def run_opts(args, model, tokenizer, embedding_table):
    # output_target = " Dan Sullivan"
    output_targets = load_outputs(f'data/{args.filename}')
    print(f'{len(output_targets)} output targets loaded')
    successes = [] 
    for i, output_target in enumerate(tqdm(output_targets, total=len(output_targets))):
        prompt_toks = run_arca(args, model, tokenizer, embedding_table, output_str=output_target)
        if prompt_toks is None:
            print(f'{i} - No prompt found')
            successes.append(0)
        else:
            prompt = tokenizer.decode(prompt_toks) 
            print(f'{i} - prompt: \t', prompt)
            successes.append(1)
    print(f'\n--- Success rate: {sum(successes) / len(successes)} ---')


if __name__ == '__main__':
    args = parse_args()
    model, tokenizer = get_model_and_tokenizer(args)
    embedding_table = get_raw_embedding_table(model)
    print('Model and tokenizer loaded')
    run_opts(args, model, tokenizer, embedding_table)
