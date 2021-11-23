import os
import torch
import json
from tqdm import tqdm
from collections import Counter, defaultdict


class FeatureReader(object):
    def __init__(self, data_path) -> None:
        self.data = torch.load(data_path)

    def read(self, split='train'):
        return self.data[split]


class TextReader(object):
    'read text feature'
    """read and store DocRED data"""
    def __init__(self, data_dir, save_dir, tokenizer) -> None:
        self.data_dir = data_dir
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        with open(os.path.join(data_dir, 'rel_info.json')) as fp:
            self.rel2info = json.load(fp)
        self.id2rel = sorted(list(self.rel2info.keys()))
        self.rel2id = {r: i for i, r in enumerate(self.id2rel)}

        self.data_paths = {
            'train': os.path.join(data_dir, 'train_annotated.json'),
            'dist': os.path.join(data_dir, 'train_distant.json'),
            'dev': os.path.join(data_dir, 'dev.json'),
            'test': os.path.join(data_dir, 'test.json')
        }
        self.bin_paths = {
            'train': os.path.join(save_dir, 'train.pth'),
            'dist': os.path.join(save_dir, 'dist.pth'),
            'dev': os.path.join(save_dir, 'dev.pth'),
            'test': os.path.join(save_dir, 'test.pth')
        }

        self.tokenizer = tokenizer

    def read(self, split='train'):
        bin_path = self.bin_paths[split]
        if os.path.exists(bin_path):
            return torch.load(bin_path)
        else:
            features = self.read_raw(split)
            torch.save(features, bin_path)
            return features

    def read_raw(self, split='train', max_seq_length=1024):
        with open(self.data_paths[split]) as fp:
            data = json.load(fp)

        features = []

        for item in tqdm(data, desc='reading raw data'):
            sents = []
            sent_map = []

            entities = item['vertexSet']
            entity_start, entity_end = [], []
            for entity in entities:
                types = []
                for mention in entity:
                    sent_id = mention["sent_id"]
                    pos = mention["pos"]
                    entity_start.append((sent_id, pos[0],))
                    entity_end.append((sent_id, pos[1] - 1,))

            for i_s, sent in enumerate(item['sents']):
                new_map = {}
                for i_t, token in enumerate(sent):
                    tokens_wordpiece = self.tokenizer.tokenize(token)
                    if (i_s, i_t) in entity_start:
                        tokens_wordpiece = ["*"] + tokens_wordpiece
                    if (i_s, i_t) in entity_end:
                        tokens_wordpiece = tokens_wordpiece + ["*"]
                    new_map[i_t] = len(sents)
                    sents.extend(tokens_wordpiece)
                new_map[i_t + 1] = len(sents)
                sent_map.append(new_map)

            entity_pos = []
            for e in entities:
                entity_pos.append([])
                for m in e:
                    start = sent_map[m["sent_id"]][m["pos"][0]]
                    end = sent_map[m["sent_id"]][m["pos"][1]]
                    entity_pos[-1].append((start, end,))

            labels = torch.zeros(len(entities), len(entities), len(self.rel2id), dtype=torch.bool)
            if 'labels' in item:
                for fact in item['labels']:
                    labels[fact['h'], fact['t'], self.rel2id[fact['r']]] = 1

            sents = sents[:max_seq_length - 2]
            input_ids = self.tokenizer.convert_tokens_to_ids(sents)
            input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)

            features.append({
                'input_ids': input_ids,
                'entity_pos': entity_pos,
                'title': item['title'],
                'N': len(entities),
                'labels': labels.to_sparse()
            })

        return features

    def get_prior(self, split='train'):
        train_data = self.read(split)
        total = 0.
        pos = torch.zeros([len(self.rel2id)])
        for f in tqdm(train_data):
            labels = f['labels'].float().to_dense()
            pos += labels.sum(dim=(0,1))
            total += labels.size(0)**2
        return pos / total


class ERuleReader(object):
    'read text feature'
    """read and store DocRED data"""
    def __init__(self, data_dir, save_dir, max_step=3) -> None:
        self.data_dir = data_dir
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.rel2id = {k: v-1 for k,v in json.load(open(os.path.join(data_dir, 'meta/rel2id.json'))).items()}
        self.id2rel = {k:v for v, k in self.rel2id.items()}
        self.R = len(self.rel2id) - 1
        self.type2id = json.load(open(os.path.join(data_dir, 'meta/ner2id.json')))
        self.id2type = {k:v for v, k in self.type2id.items()}

        self.data_paths = {
            'rtrain': os.path.join(data_dir, 'rtrain.json'),
            'train': os.path.join(data_dir, 'train_annotated.json'),
            'dist': os.path.join(data_dir, 'train_distant.json'),
            'dev': os.path.join(data_dir, 'dev.json'),
            'test': os.path.join(data_dir, 'test.json')
        }
        self.bin_paths = {
            'rtrain': os.path.join(save_dir, 'cooccur-rtrain.pth'),
            'train': os.path.join(save_dir, 'cooccur-train.pth'),
            'dist': os.path.join(save_dir, 'cooccur-dist.pth'),
            'dev': os.path.join(save_dir, 'cooccur-dev.pth'),
            'test': os.path.join(save_dir, 'cooccur-test.pth')
        }
        self.max_step = max_step

    def read(self, split='train'):
        bin_path = self.bin_paths[split]
        if os.path.exists(bin_path):
            return torch.load(bin_path)
        else:
            features = self.read_raw(split)
            torch.save(features, bin_path)
            return features

    def read_raw(self, split='train'):
        """count co-occurence info"""
        max_step = self.max_step
        r2epair = self.get_r2epair()
        rule_counter = {(i, h, t): Counter() for i in range(self.R) for (h, t) in r2epair[i]}

        with open(self.data_paths[split]) as fp:
            data = json.load(fp)

        for item in tqdm(data, desc='reading raw data'):
            entities = item['vertexSet']
            entity_types = [self.type2id[e[0]['type']] for e in entities]

            paths = {}
            meta_paths = {1: paths}

            for fact in item['labels']:
                h, t, r = fact['h'], fact['t'], self.rel2id[fact['r']]
                if h not in paths:
                    paths[h] = {t: [([r], [t])]}
                elif t not in paths[h]:
                    paths[h][t] = [([r], [t])]
                else:
                    paths[h][t].append(([r], [t]))

                if t not in paths:
                    paths[t] = {h: [([r + self.R], [h])]}
                elif h not in paths[t]:
                    paths[t][h] = [([r + self.R], [h])]
                else:
                    paths[t][h].append(([r + self.R], [h]))

            for step in range(2, max_step + 1):
                prev_paths = meta_paths[step - 1]
                paths = {}
                for h in prev_paths:
                    for inode, prev_chain in prev_paths[h].items():
                        if inode in meta_paths[1]:
                            for t, rs in meta_paths[1][inode].items():
                                if h == t:
                                    continue
                                new_chain = append_chain(prev_chain, rs)
                                if not new_chain:
                                    continue
                                if h not in paths:
                                    paths[h] = {t: new_chain}
                                elif t not in paths[h]:
                                    paths[h][t] = new_chain
                                else:
                                    paths[h][t].extend(new_chain)
                meta_paths[step] = paths

            for h in meta_paths[1]:
                for t, rs in meta_paths[1][h].items():
                    c_meta_paths = set()
                    for step in range(1, max_step + 1):
                        if h in meta_paths[step] and t in meta_paths[step][h]:
                            for path in meta_paths[step][h][t]:
                                c_meta_paths.add(tuple(path[0]))
                    for r in rs:
                        if r[0][0] >= self.R:
                            continue
                        triple = (r[0][0], entity_types[h], entity_types[t])
                        rule_counter[triple].update(c_meta_paths)
        
        triples = []
        triple2rules = {}
        triple2probs = {}
        lens = [len(epair) for epair in r2epair]
        for ri, epairs in enumerate(r2epair):
            for epair in epairs:
                triple = (ri, epair[0], epair[1])
                total = sum(rule_counter[triple].values())
                rules, probs = [], []
                for rule in rule_counter[triple]:
                    rules.append(rule)
                    probs.append(rule_counter[triple][rule] / total)

                triples.append(triple)
                triple2rules[triple] = rules
                triple2probs[triple] = probs

        features = {
            'triples': triples,
            'sections': lens,
            'triple2rules': triple2rules,
            'triple2probs': triple2probs,
        }

        return features

    def get_r2epair(self):
        r2epair = [[] for _ in range(len(self.rel2id)-1)]
        with open(self.data_paths['train']) as fp:
            data = json.load(fp)
        for item in data:
            entities = item['vertexSet']
            entity_types = [self.type2id[e[0]['type']] for e in entities]

            for fact in item['labels']:
                h, t, r = entity_types[fact['h']], entity_types[fact['t']], self.rel2id[fact['r']]
                if (h,t) not in r2epair[r]:
                    r2epair[r].append((h, t))

        return r2epair

    def get_epair2r(self):
        e_pair2r = torch.zeros(len(self.type2id), len(self.type2id), len(self.rel2id)-1).bool()
        with open(self.data_paths['train']) as fp:
            data = json.load(fp)
        for item in data:
            entities = item['vertexSet']
            entity_types = [self.type2id[e[0]['type']] for e in entities]

            for fact in item['labels']:
                h, t, r = fact['h'], fact['t'], self.rel2id[fact['r']]
                e_pair2r[entity_types[h], entity_types[t], r] = 1
        print(e_pair2r.size(), e_pair2r.sum())
        return e_pair2r

    def get_type_mask(self, triples, sections, split='train'):
        ntypes = len(self.type2id)
        rpair2id = [{} for _ in sections]
        tid = 0
        for section in sections:
            for sid in range(section):
                r, e1, e2 = triples[tid]
                rpair2id[r][(e1, e2)] = sid
                tid += 1

        triple2sid = torch.CharTensor(ntypes, ntypes, self.R).fill_(-1)
        for ei in range(ntypes):
            for ej in range(ntypes):
                for r in range(self.R):
                    triple2sid[ei, ej, r] = rpair2id[r].get((ei, ej), -1)

        with open(self.data_paths[split]) as fp:
            data = json.load(fp)

        type_masks = []
        for item in data:
            entities = item['vertexSet']
            N = len(entities)
            entity_types = torch.tensor([self.type2id[e[0]['type']] for e in entities])
            type_indices = (entity_types.unsqueeze(1).repeat(1, N), entity_types.unsqueeze(0).repeat(N, 1))
            type_mask = triple2sid[type_indices[0], type_indices[1]]
            type_masks.append(type_mask)
        
        return type_masks

    def get_dist(self, split='train'):
        with open(self.data_paths[split]) as fp:
            data = json.load(fp)

        dists = []
        for item in tqdm(data, desc='reading raw data'):
            entities = item['vertexSet']
            N = len(entities)
            entities_pos = []
            for entity in entities:
                s = entity[0]['pos'][0]
                e = entity[0]['pos'][1]
                entities_pos.append([s, e])
            dist = torch.zeros(N, N)
            for h in range(N):
                for t in range(N):
                    sh, eh = entities_pos[h]
                    st, et = entities_pos[t]
                    dist[h,t] = min(abs(sh - et), abs(st - eh))
            dists.append(dist)
        return dists

                            
def append_chain(chains, rs):
    ret = []
    for chain, chain_nodes in chains:
        for r, rnode in rs:
            if rnode[0] not in chain_nodes:
                ret.append((chain + r, chain_nodes + rnode))
    return ret



if __name__ == "__main__":
    reader = ERuleReader('../kbp-benchmarks/DWIE/data/docred-style', 'data/DWIE-erules')
    reader.get_dist('train')
