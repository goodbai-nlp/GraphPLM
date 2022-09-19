import os
from pathlib import Path

import torch
try:
    from torch.cuda.amp import autocast
    autocast_available = True
except ImportError:
    class autocast:
        def __init__(self, enabled=True): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_value, exc_traceback): pass
    autocast_available = False

from torch.cuda.amp.grad_scaler import GradScaler
import transformers

from spring_amr import ROOT
from spring_amr.dataset import reverse_direction
from spring_amr.optim import RAdam
from spring_amr.evaluation import write_predictions, compute_smatch, predict_amrs, predict_sentences, compute_bleu
from spring_amr.utils import instantiate_model_and_tokenizer, instantiate_loader, instantiate_tokenizer

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import ModelCheckpoint, global_step_from_engine


def preprocess(data_path, data_output_path, checkpoint=None, direction='amr', split_both_decoder=False, fp16=False):

    assert direction in ('amr', 'text', 'both')

    tokenizer = instantiate_tokenizer(
        config['model'],
        checkpoint=checkpoint,
        additional_tokens_smart_init=config['smart_init'],
        dropout=config['dropout'],
        attention_dropout=config['attention_dropout'],
        from_pretrained=config['warm_start'],
        init_reverse=split_both_decoder,
        penman_linearization=config['penman_linearization'],
        collapse_name_ops=config['collapse_name_ops'],
        use_pointer_tokens=config['use_pointer_tokens'],
        raw_graph=config.get('raw_graph', False)
    )

    train_gold_path = ROOT / f'../data/{config["cate"]}/train-gold.amr'
    train_loader = instantiate_loader(
        data_path,
        tokenizer,
        batch_size=config['batch_size'],
        evaluation=True, out=train_gold_path,
        use_recategorization=config['use_recategorization'],
        remove_longer_than=config['remove_longer_than'],
        remove_wiki=config['remove_wiki'],
        dereify=config['dereify'],
        output_path=data_output_path,
        data_cate=config["cate"],
    )

    #dev_gold_path = ROOT / f'../data/{config["cate"]}/val-gold.amr'
    #dev_loader = instantiate_loader(
    #    config['dev'],
    #    tokenizer,
    #    batch_size=config['batch_size'],
    #    evaluation=True, out=dev_gold_path,
    #    use_recategorization=config['use_recategorization'],
    #    remove_wiki=config['remove_wiki'],
    #    dereify=config['dereify'],
    #    type_path="val",
    #    data_cate=config["cate"],
    #)
    #test_gold_path = ROOT / f'../data/{config["cate"]}/test-gold.amr'
    #
    #test_loader = instantiate_loader(
    #    config['test'],
    #    tokenizer,
    #    batch_size=config['batch_size'],
    #    evaluation=True, out=test_gold_path,
    #    use_recategorization=config['use_recategorization'],
    #    type_path="test",
    #    data_cate=config["cate"],
    #)


if __name__ == '__main__':

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    import yaml

    import wandb

    parser = ArgumentParser(
        description="Trainer script",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--direction', type=str, default='amr', choices=['amr', 'text', 'both'],
        help='Train a uni- (amr, text) or bidirectional (both).')
    parser.add_argument('--split-both-decoder', action='store_true')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/sweeped.yaml',
        help='Use the following config for hparams.')
    parser.add_argument('--checkpoint', type=str,
        help='Warm-start from a previous fine-tuned checkpoint.')
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    parser.add_argument('--fp16', action='store_true')
    args, unknown = parser.parse_known_args()

    if args.fp16 and autocast_available:
        raise ValueError('You\'ll need a newer PyTorch version to enable fp16 training.')

    with args.config.open() as y:
        config = yaml.load(y, Loader=yaml.FullLoader)

    with open('wiki_amr_fof', 'r') as f:
        sections = [sec.strip() for sec in f]

    wiki_amr_fof = []
    wiki_amr_output_fof = []
    data_prepath = '/cephfs/user/lfsong/exp.amr_pretrain/wiki_amr_parsed/outputs/'
    data_postpath = 'val_outputs'
    data_output_prepath = '/cephfs/user/lfsong/exp.amr_pretrain/GraphPLM/spring/pretrain_inputs_eval'
    for i in range(args.start, args.end):
        cur_dir = os.path.join(data_prepath, sections[i], data_postpath)
        if os.path.isdir(cur_dir):
            cur_fof = [os.path.join(cur_dir, cur_file) for cur_file in os.listdir(cur_dir) if cur_file.endswith('.txt')]
            wiki_amr_fof.append(cur_fof)
            wiki_amr_output_fof.append(os.path.join(data_output_prepath, sections[i] + '.jsonl'))

    if config['log_wandb']:
        wandb.init(
            entity="SOME-RUNS",
            project="SOME-PROJECT",
            config=config,
            dir=str(ROOT / 'runs/'))
        config = wandb.config

    print(config)

    if args.checkpoint:
        checkpoint = args.checkpoint
    else:
        checkpoint = None

    for i in range(len(wiki_amr_fof)):
        preprocess(wiki_amr_fof[i], wiki_amr_output_fof[i],
            checkpoint=checkpoint,
            direction=args.direction,
            split_both_decoder=args.split_both_decoder,
            fp16=args.fp16,
        )
