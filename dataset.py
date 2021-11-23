import torch
import numpy as np
from torch.utils.data import Dataset


class BackboneDataset(Dataset):
    def __init__(self, features, type_masks, dists):
        self.features = features
        self.type_masks = type_masks
        self.dists = dists
        assert len(features) == len(type_masks), (len(features), len(type_masks))

    def __getitem__(self, idx):
        return {**self.features[idx], **{'type_masks': self.type_masks[idx], 'dist': self.dists[idx]}}

    def __len__(self):
        return len(self.features)


def get_backbone_collate_fn(device='cpu'):

    def collate_fn(batch):
        bsz = len(batch)
        R = batch[0]['logits'].size(2)
        maxN = max(f['N'] for f in batch)
        in_train = torch.zeros([bsz, maxN, maxN, R]).bool().to(device)
        type_masks = torch.zeros([bsz, maxN, maxN, R]).char().fill_(-1).to(device)
        masks = torch.zeros([bsz, maxN, maxN]).to(device)
        labels = torch.zeros([bsz, maxN, maxN, R]).to(device)
        logits = torch.empty([bsz, maxN, maxN, 2*R+1]).fill_(-1000.).to(device)
        dists = torch.zeros([bsz, maxN, maxN]).to(device)

        for i, f in enumerate(batch):
            N = f['N']
            masks[i, :N, :N] = 1.
            labels[i, :N, :N] = f['labels']
            type_masks[i, :N, :N] = f['type_masks']
            dists[i, :N, :N] = f['dist']
            logp = torch.nn.functional.logsigmoid(f['logits']).to(device)
            logits[i, :N, :N] = torch.cat([
                logp, logp.transpose(0, 1),
                torch.empty(N, N).fill_(-1000.).masked_fill(torch.eye(N).bool(), 0.).unsqueeze(-1).to(device)
            ], dim=-1)
            in_train[i, :N, :N] = f['in_train']

        return {'logits':logits, 'labels':labels, 'masks':masks, 'in_train':in_train, 'type_masks': type_masks, 'dists': dists, 'Ns': [f['N'] for f in batch]}

    return collate_fn


class ERuleDataset(Dataset):

    def __init__(self, features, max_depth=3, N=100000):
        self.triples = features['triples']
        self.t2rules = features['triple2rules']
        self.t2probs = features['triple2probs']
        self.sections = features['sections']
        self.max_depth = max_depth
        self.R = len(self.sections)
        self.N = N

    def __getitem__(self, idx):
        ti = np.random.choice(range(len(self.triples)))
        triple = self.triples[ti]
        ci = np.random.choice(range(len(self.t2rules[triple])), p=self.t2probs[triple])
        # ci = np.random.choice(range(len(self.t2rules[triple])))
        rbody = self.t2rules[triple][ci]

        head, tail = triple[1], triple[2]
        chain = [triple[0]] + list(rbody) + (self.max_depth + 1 - len(rbody)) * [self.R * 2]

        return torch.tensor(chain), head, tail

    def __len__(self):
        return self.N

    def update(self, data, ratio=0.1):
        for triple in self.triples:
            visited = []
            for ci, rule in enumerate(data.t2rules[triple]):
                nprob = data.t2probs[triple][ci]
                if rule in self.t2rules[triple]:
                    ci_ = self.t2rules[triple].index(rule)
                    visited.append(ci_)
                    oprob = self.t2probs[triple][ci_]
                    self.t2probs[triple][ci_] = (1 - ratio) * oprob + ratio * nprob
                else:
                    visited.append(len(self.t2rules[triple]))
                    self.t2rules[triple].append(rule)
                    self.t2probs[triple].append(nprob * ratio)
                    
            for ci_ in range(len(self.t2rules[triple])):
                if ci_ not in visited:
                    self.t2probs[triple][ci_] *= (1 - ratio)


class NaiveRuleDataset(ERuleDataset):
    
    def __getitem__(self, idx):
        ti = np.random.choice(range(len(self.triples)))
        triple = self.triples[ti]
        ci = np.random.choice(range(len(self.t2rules[triple])))
        rbody = self.t2rules[triple][ci]

        head, tail = triple[1], triple[2]
        chain = [triple[0]] + list(rbody) + (self.max_depth + 1 - len(rbody)) * [self.R * 2]

        return torch.tensor(chain), head, tail


class PRuleDataset(Dataset):
    def __init__(self, erule_data, p_samples) -> None:
        self.triples = erule_data.triples
        self.t2rules = {}
        self.t2probs = {}

        # approximate posterior
        for ti, counter in enumerate(p_samples):
            triple = self.triples[ti]
            self.t2rules[triple] = []
            self.t2probs[triple] = []
            total = sum(counter.values())
            for rule, count in counter.items():
                self.t2rules[triple].append(rule)
                self.t2probs[triple].append(float(count) / total)

        self.max_depth = erule_data.max_depth
        self.R = erule_data.R
        self.N = erule_data.N

    def __getitem__(self, idx):
        ti = np.random.choice(range(len(self.triples)))
        triple = self.triples[ti]
        ci = np.random.choice(range(len(self.t2rules[triple])), p=self.t2probs[triple])
        rbody = self.t2rules[triple][ci]

        head, tail = triple[1], triple[2]
        chain = [triple[0]] + list(rbody) + (self.max_depth + 1 - len(rbody)) * [self.R * 2]

        return torch.tensor(chain), head, tail

    def __len__(self):
        return self.N


class MixRuleDataset(Dataset):
    def __init__(self, datasets, probs, N=100000) -> None:
        self.datasets = datasets
        self.probs = probs
        self.N = N

    def __getitem__(self, idx):
        di = np.random.choice(range(len(self.datasets)), p=self.probs)
        return self.datasets[di][idx]

    def __len__(self):
        return self.N
