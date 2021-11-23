import torch
import os
from argparse import ArgumentParser
from logire import LogiRE, RelationExtractor
from dataset import BackboneDataset, get_backbone_collate_fn
from torch.utils.data import DataLoader


def main():
    parser = ArgumentParser()
    parser.add_argument('--mode', default='train')
    parser.add_argument('--save_dir', default='logire-save')
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--Ns', type=int, default=50, help="size of the latent rule set")
    parser.add_argument('--num_epochs', type=int, default=50, help="number of training epochs for the relation extractor")
    parser.add_argument('--warmup_ratio', type=float, default=0.06)
    parser.add_argument('--rel_num', type=int, default=65, help="number of relation types")
    parser.add_argument('--ent_num', type=int, default=10, help='number of entity types')
    parser.add_argument('--n_iters', type=int, default=10, help='number of iterations')
    parser.add_argument('--max_depth', type=int, default=3, help='max depth of the rules')
    parser.add_argument('--data_dir', default='../kbp-benchmarks/DWIE/data/docred-style')
    parser.add_argument('--backbone_path', default="data/dwie-atlop.dump")
    parser.add_argument('--rule_path', default='data/dwie.grules.json')

    args = parser.parse_args()

    if args.mode == 'train':
        logire = LogiRE(args)
        logire.EM_optimization()
    elif args.mode == 'test':
        logire = LogiRE(args)
        dev_ret, test_ret = logire.evaluate_base()
        print('#' * 100 + '\n# Evaluating Backbone\n' + '#' * 100)
        print('dev ', dev_ret)
        print('test', test_ret)

        collate_fn = get_backbone_collate_fn(0)
        dev_data = BackboneDataset(logire.re_reader.read('dev'), logire.type_masks['dev'], logire.dists['dev'])
        dev_loader = DataLoader(dev_data, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn)
        test_data = BackboneDataset(logire.re_reader.read('test'), logire.type_masks['test'], logire.dists['test'])
        test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn)

        print('#' * 100 + '\n# Evaluating LogiRE\n' + '#' * 100)
        for iter_i in range(args.n_iters + 1):
            print('-'*45 + f'Iter {iter_i}' + '-'*50)
            save_path = os.path.join(args.save_dir, f'scorer-{iter_i}.pt')
            model = RelationExtractor(torch.load(save_path))
            dev_ret = logire.evaluate_relation_extractor(model, dev_loader)
            print('dev ', dev_ret)
            test_ret = logire.evaluate_relation_extractor(model, test_loader, dev_ret['theta'])
            print('test', test_ret)
    else:
        raise ValueError(f'Unknown mode {args.mode}')


if __name__ == "__main__":
    main()