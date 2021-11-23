import torch
import torch.nn as nn
import math
import os

from torch.nn.modules.loss import BCEWithLogitsLoss
from evaluation import Evaluator
from collections import Counter
from tqdm import tqdm
from torch.distributions import Categorical
from reader import ERuleReader, FeatureReader
from dataset import ERuleDataset, PRuleDataset, BackboneDataset, get_backbone_collate_fn, NaiveRuleDataset, MixRuleDataset
from transformers.optimization import AdamW, get_polynomial_decay_schedule_with_warmup
from torch.utils.data import DataLoader


class LogiRE():
    """The LogiRE framework for doc-level RE"""
    def __init__(self, args) -> None:
        self.args = args
        self.rule_generator = None
        self.relation_extractor = None
        self.n_iters = args.n_iters
        self.rel_num = args.rel_num
        self.ent_num = args.ent_num
        self.max_depth = args.max_depth

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        self.save_dir = args.save_dir

        self.logfile = os.path.join(self.save_dir, 'log')
        
        self.rg_reader = ERuleReader(
            args.data_dir,
            os.path.join(self.save_dir, 'cooccur-data'),
            max_step=self.max_depth
        )
        self.cooccur_data = ERuleDataset(self.rg_reader.read(), max_depth=self.max_depth)
        self.naive_data = NaiveRuleDataset(self.rg_reader.read(), max_depth=self.max_depth)

        self.re_reader = FeatureReader(args.backbone_path)

        rule_data = self.rg_reader.read()
        # possible candidate triples and number of triples for each relation
        self.triples, self.sections = rule_data['triples'], rule_data['sections']
        self.type_masks = {
            split: self.rg_reader.get_type_mask(self.triples, self.sections, split) for split in ['train', 'dev', 'test']
        }
        self.dists = {
            split: self.rg_reader.get_dist(split) for split in ['train', 'dev', 'test']
        }

        self.evaluator = Evaluator(args)

    def logging(self, msg):
        print(msg)
        with open(self.logfile, 'a+') as f_log:
            f_log.write(msg + '\n')
    
    def initialize(self):
        # initialize rule generator
        self.logging('#' * 100 + '\n' + '# Training Rule Generator\n' + '#' * 100)
        self.pretrain_rule_generator()
        # initialize relation extractor
        self.logging('#' * 100 + '\n' + '# Training Relation Extractor\n' + '#' * 100)
        self.train_relation_extractor()

    def pretrain_rule_generator(self):
        self.rule_generator = RuleGenerator(max_depth=self.max_depth, rel_num=self.rel_num, ent_num=self.ent_num).to(0)

        rg_path = os.path.join(self.save_dir, 'rg-0.pt')
        if os.path.exists(rg_path):
            self.logging('loading from checkpoint')
            self.rule_generator.load_state_dict(torch.load(rg_path))
            return

        # mix with naive sampling for more exploration
        mix_data = MixRuleDataset([self.cooccur_data, self.naive_data], [0.9, 0.1])
        loader = DataLoader(mix_data, batch_size=32, num_workers=2)
        
        self.rule_generator.train_model(loader, 50)
        torch.save(self.rule_generator.state_dict(), rg_path)

    def train_relation_extractor(self, save_path='scorer-0.pt'):
        save_path = os.path.join(self.save_dir, save_path)
        if os.path.exists(save_path):
            self.logging('loading from checkpoint')
            self.relation_extractor = RelationExtractor(torch.load(save_path))
            return

        chains, scores, counts, c_sections = self.rule_generator.sample_rules(self.triples, self.args.Ns)
        rule_scorer = RuleScorer(self.triples, chains, scores, counts, self.sections, c_sections).to(0)
        model = RelationExtractor(rule_scorer)
        self.relation_extractor = model
        
        collate_fn = get_backbone_collate_fn(0)
        train_data = BackboneDataset(self.re_reader.read('train'), self.type_masks['train'], self.dists['train'])
        train_loader = DataLoader(train_data, batch_size=self.args.train_batch_size, shuffle=True, collate_fn=collate_fn)
        dev_data = BackboneDataset(self.re_reader.read('dev'), self.type_masks['dev'], self.dists['dev'])
        dev_loader = DataLoader(dev_data, batch_size=self.args.test_batch_size, shuffle=False, collate_fn=collate_fn)
        test_data = BackboneDataset(self.re_reader.read('test'), self.type_masks['test'], self.dists['test'])
        test_loader = DataLoader(test_data, batch_size=self.args.test_batch_size, shuffle=False, collate_fn=collate_fn)

        opt = AdamW(model.parameters(), lr=5e-3)

        total_steps = len(train_loader) * self.args.num_epochs
        warmup_steps = int(total_steps * self.args.warmup_ratio)
        self.logging("Total steps: {}".format(total_steps))
        self.logging("Warmup steps: {}".format(warmup_steps))
        scheduler = get_polynomial_decay_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps, lr_end=1e-4)
        steps = 0

        best_score = -1. 
        for ei in range(self.args.num_epochs):
            for batch in tqdm(train_loader, ncols=80, desc=f'RE Training epoch {ei+1}'):
                opt.zero_grad()
                loss, _, _ = model(batch)
                loss.backward()
                opt.step()
                scheduler.step()
                steps += 1

            eval_dict = self.evaluate_relation_extractor(model, dev_loader)
            dev_score = eval_dict['ignf1']
            self.logging(f'Epoch {ei+1} dev: {eval_dict}')
            if dev_score > best_score:
                torch.save(model.scorer, save_path)
                best_score = dev_score
                eval_dict = self.evaluate_relation_extractor(model, test_loader, eval_dict['theta'])
                self.logging(f'Epoch {ei+1} test: {eval_dict}')

    def evaluate_base(self):
        dev_data = self.re_reader.read('dev')
        test_data = self.re_reader.read('test')
        self.evaluator.reset()
        for item in dev_data:
            self.evaluator.add_item(item['logits'].sigmoid(), item['labels'], item['in_train'])
        dev_ret = self.evaluator.get_ret()
        theta = dev_ret['theta']

        self.evaluator.reset()
        for item in test_data:
            self.evaluator.add_item(item['logits'].sigmoid(), item['labels'], item['in_train'])
        test_ret = self.evaluator.get_ret(theta)
        return dev_ret, test_ret

    @torch.no_grad()
    def evaluate_relation_extractor(self, model, loader, theta=None):
        model.eval()
        self.evaluator.reset()
        for batch in loader:
            _, logits, _ = model(batch)
            for logits_i, labels_i, intrain_i, Ni in zip(logits, batch['labels'], batch['in_train'], batch['Ns']):
                self.evaluator.add_item(logits_i[:Ni, :Ni].sigmoid().detach().cpu(), labels_i[:Ni, :Ni].detach().cpu(), intrain_i[:Ni, :Ni].detach().cpu())
        model.train()
        return self.evaluator.get_ret(theta=theta)

    def E_step(self):
        """approximate the posterior of each rule"""
        collate_fn = get_backbone_collate_fn(0)
        train_batch_size = 4
        train_data = BackboneDataset(self.re_reader.read('train'), self.type_masks['train'], self.dists['train'])
        train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)
        posterior_samples = [Counter() for _ in self.triples]
        for batch in tqdm(train_loader, ncols=80, desc='posterior inference'):
            posterior_samplesi = self.relation_extractor.sample_rules_by_posterior(batch)
            for counter_t, counter_ti in zip(posterior_samples, posterior_samplesi):
                counter_t.update(counter_ti)

        posterior_data = PRuleDataset(self.cooccur_data, posterior_samples)
        return posterior_data

    def M_step(self, posterior_data, rg_save_path, scorer_save_path):
        """update the rule generator and relation extractor according to the approximated posterior"""
        # train rule generator
        # global sampling + momenta update instead of sampling + posterior inference for each instance for better optimization efficiency 
        self.cooccur_data.update(posterior_data, 0.1)
        loader = DataLoader(self.cooccur_data, batch_size=32, num_workers=2)
        self.rule_generator.train_model(loader, 20)
        torch.save(self.rule_generator.state_dict(), os.path.join(self.save_dir, rg_save_path))
        # train relation extractor
        self.train_relation_extractor(scorer_save_path)
        
    def EM_optimization(self):
        self.initialize()
        for iter_i in range(self.n_iters):
            self.logging('#' * 100 + f'\n# Iter {iter_i+1} Optimization\n' + '#' * 100)
            posterior_data = self.E_step()
            torch.save(posterior_data, os.path.join(self.save_dir, f'psamples-{iter_i+1}.pt'))
            self.M_step(posterior_data, f'rg-{iter_i+1}.pt', f'scorer-{iter_i+1}.pt')


@torch.no_grad()
def evaluate(model, data, tag='dev', theta=None):
    model.eval()
    scores, labels, in_trains = [], [], []
    for batch in tqdm(data, ncols=50):
        _, probs, ys, _ = model(batch)
        scores.append(probs.reshape(-1))
        labels.append(ys.reshape(-1))
        in_train = batch['in_train'].masked_select(batch['masks'].bool().unsqueeze(-1)).reshape(-1)
        in_trains.append(in_train)

    scores = torch.cat(scores)
    labels = torch.cat(labels)
    in_trains = torch.cat(in_trains)

    if theta is None:
        sorted_scores, sorted_indices = scores.sort(descending=True)
        sorted_labels = labels[sorted_indices]
        sorted_intrains = in_trains[sorted_indices] * sorted_labels
        cum_intrains = sorted_intrains.cumsum(dim=0)
        true = sorted_labels.cumsum(dim=0)
        positive = torch.arange(len(true)).to(true) + 1
        prec = true / positive
        prec_ign = (true - cum_intrains) / (positive - cum_intrains)
        rec = true / labels.sum()
        f1s = 2 * prec * rec / (prec + rec)
        f1s[f1s.isnan()] = 0.
        _, maxi = f1s.max(dim=0)
        theta = sorted_scores[maxi]

    predicted = scores > theta
    labels = labels > 0
    prec = (predicted & labels).sum() / predicted.sum()
    rec = (predicted & labels).sum() / labels.sum()
    f1 = 2 * prec * rec / (prec + rec)

    ign_rec = (predicted & labels & (~in_trains)).sum() / (labels & (~in_trains)).sum()
    ignf1 = 2 * prec * ign_rec / (prec + ign_rec)

    model.train()

    return f1, {
        f'{tag}_f1_ign': ignf1 * 100., f'{tag}_f1': f1 * 100., f'{tag}_theta': theta
    }


class RuleGenerator(nn.Module):
    def __init__(self, hidden_size: int=256, max_depth: int=3, rel_num: int=65, ent_num: int=10, layer_num: int=2):
        super(RuleGenerator, self).__init__()
        self.R = rel_num
        self.E = ent_num
        self.hidden_size = hidden_size
        self.rel_emb = nn.Embedding(rel_num * 2 + 1, hidden_size)
        self.ent_emb = nn.Embedding(ent_num, hidden_size)
        self.transformer = nn.Transformer(hidden_size, num_encoder_layers=layer_num, num_decoder_layers=layer_num)
        self.proj = nn.Linear(hidden_size, 2 * rel_num + 1)
        self.max_depth = max_depth
        self.loss_fnt = nn.CrossEntropyLoss(reduction='mean')

    def get_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, chains, heads, tails):
        bsz, L = chains.size()
        chain_embs = self.rel_emb(chains)  # [N, L, h]
        src_emb = torch.stack([self.ent_emb(heads), chain_embs[:, 0], self.ent_emb(tails)], dim=0)  # [3, N, h]
        tgt_in = torch.cat([
            torch.zeros(bsz, 1, self.hidden_size).to(src_emb),
            chain_embs[:, 1:]
        ], dim=1).transpose(0, 1)  # [L, N, h]
        tgt_mask = self.get_mask(L).to(src_emb)
        tgt_out = self.transformer(src_emb, tgt_in, tgt_mask=tgt_mask).transpose(0, 1)
        logits = self.proj(tgt_out)

        return logits

    def compute_loss(self, chains, heads, tails):
        inputs = chains[:, :-1]
        targets = chains[:, 1:]
        logits = self.forward(inputs, heads, tails)
        loss = self.loss_fnt(logits.reshape(-1, 2*self.R+1), targets.reshape(-1))

        return loss

    @torch.no_grad()
    def sample_rules(self, triples, N=10):
        self.eval()
        all_chains = []
        all_scores = []
        all_counts = []
        sections = []
        for triple in triples:
            heads = torch.LongTensor([triple[1]] * N).to(0)
            tails = torch.LongTensor([triple[2]] * N).to(0)
            rels = torch.LongTensor([triple[0]] * N).to(0)
            src_emb = torch.stack([self.ent_emb(heads), self.rel_emb(rels), self.ent_emb(tails)])
            tgt_in = torch.zeros(1, N, self.hidden_size).to(src_emb)
            chains = []
            scores = torch.zeros(N).to(0)
            for i in range(self.max_depth):
                tgt_mask = self.get_mask(i + 1).to(src_emb)
                tgt_out = self.transformer(src_emb, tgt_in, tgt_mask=tgt_mask)[-1]
                probs = self.proj(tgt_out).softmax(dim=-1)  # [N, V]
                dist = Categorical(probs=probs)
                next_rels = dist.sample() # [N]

                scores += torch.gather(probs, dim=-1, index=next_rels.unsqueeze(-1)).squeeze().log()
                chains.append(next_rels)

                tgt_in = torch.cat([
                    torch.zeros(N, 1, self.hidden_size).to(src_emb),
                    self.rel_emb(torch.stack(chains, dim=-1))
                ], dim=1).transpose(0, 1)
                

            chains = torch.stack(chains, dim=-1)  # [N, L]
            reduced_chains, indices, counts = chains.unique(sorted=False, return_inverse=True, return_counts=True, dim=0)
            reduced_scores = torch.zeros(len(reduced_chains)).to(0)
            for i, score in zip(indices, scores):
                reduced_scores[i] = score

            all_chains.append(reduced_chains)
            all_scores.append(reduced_scores)
            all_counts.append(counts)
            sections.append(len(reduced_chains))

        self.train()

        print(len(sections), sum(sections))
        return torch.cat(all_chains, dim=0), torch.cat(all_scores), torch.cat(all_counts), sections

        
    def train_model(self, data_iter, num_epochs=300, lr=1e-3):
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 1, 0.9)

        total_loss = 0
        for ei in range(num_epochs):
            for batch in tqdm(data_iter, ncols=80, desc='train rule generator: '):
                chains, heads, tails = batch
                loss = self.compute_loss(chains.to(0), heads.to(0), tails.to(0))
                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += loss.item()
            print(f'train {ei}\t loss {total_loss / len(data_iter)}')
            total_loss = 0
            scheduler.step()


class RelationExtractor(nn.Module):
    def __init__(self, scorer):
        super(RelationExtractor, self).__init__()
        self.scorer = scorer
        self.loss_fnt = nn.BCEWithLogitsLoss(reduction='none')
        # self.loss_fnt = nn.BCELoss(reduction='mean')

    def forward(self, inputs):
        base_logits, labels, masks, type_masks, Ns = inputs['logits'], inputs['labels'], inputs['masks'], inputs['type_masks'], inputs['Ns']
        logits = self.scorer(base_logits, type_masks)
        bsz, _, _, R = logits.size()
        loss = (self.loss_fnt(logits.reshape(-1, R), labels.reshape(-1, R)) * masks.reshape(-1).unsqueeze(-1)).sum() / masks.sum()
        labels = labels.masked_select(masks.bool().unsqueeze(-1))
        probs = logits.sigmoid().masked_select(masks.bool().unsqueeze(-1))
        # print(logits.size(), base_logits.size())
        # probs = base_logits[:, :, :, :65].exp().masked_select(masks.bool().unsqueeze(-1))
        pv = (probs * labels).sum() / labels.sum()
        nv = (probs * (1-labels)).sum() / (1-labels).sum()

        return loss, logits, {'p_text': pv, 'n_text': nv}

    def sample_rules_by_posterior(self, inputs):
        base_logits, labels, masks, type_masks = inputs['logits'], inputs['labels'], inputs['masks'], inputs['type_masks']
        labels = labels * 2. - 1  # convert to -1 +1 labels
        posterior_counter = self.scorer.get_posterior(base_logits, type_masks, labels)
        return posterior_counter


class RuleScorer(nn.Module):
    def __init__(self, triples, rules, rule_scores, rule_counts, t_sections, c_sections, Ns=50):
        """
        params:
            triples
            rules [Nc, L]
            rule_scores [Nc]
            rule_counts [Nc]
            t_sections: list of numbers of triples for each rule
            c_sections: list of numbers of chains for each triple
        """

        super(RuleScorer, self).__init__()

        # register buffer so that these variables will be saved in the state_dict
        self.register_buffer('triples', torch.tensor(triples))
        self.register_buffer('rules', torch.tensor(rules))
        self.register_buffer('rule_scores', torch.tensor(rule_scores))
        self.register_buffer('rule_counts', torch.tensor(rule_counts))
        self.register_buffer('t_sections', torch.tensor(t_sections))
        self.register_buffer('c_sections', torch.tensor(c_sections))
        self.Ns = Ns

        self.path_scorer = PathScorer()
        self.weights = nn.ParameterList([nn.Parameter(torch.Tensor(Ni, 1)) for Ni in c_sections])
        self.biases = nn.ParameterList([nn.Parameter(torch.Tensor(1)) for _ in c_sections])
        self.reset_parameters()

    @property
    def Nt(self):
        """number of triples"""
        return len(self.c_sections)

    @property
    def Nc(self):
        """number of chains"""
        return int(self.c_sections.sum())

    def reset_parameters(self):
        for weight in self.weights:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        for bias in self.biases:
            nn.init.uniform_(bias, -1, 1)

    def forward(self, transitions, type_mask):
        """
        params:
            transitions: [B, N, N, 2R+1]
            type_mask: [B, N, N, R]
        """
        B, N, _, R = type_mask.size()
        out = torch.empty(B, N, N, R).fill_(-1000.).to(transitions)

        scores = self.path_scorer(transitions, self.rules).exp()  # [B, N, N, Nc]
        scores_split = scores.split(self.c_sections.cpu().numpy().tolist(), dim=-1)  # List[[B, N, N, Nc_i], ...]

        ci = 0
        for i in range(R):
            if self.t_sections[i] == 0:
                continue
            # the ti-th triple of the i-th target relation
            for ti in range(self.t_sections[i]):
                scores_i = (scores_split[ci + ti].unsqueeze(-1) * self.weights[ci + ti][None, None, None, ...]).sum(dim=(-2)) \
                + self.biases[ci + ti][None, None, None, ...]  # [B, N, N, 1]
                mask_i = (type_mask[:, :, :, i] == ti)
                # print(mask_i.device, scores_i.device())
                out[:, :, :, i].masked_scatter_(mask_i, scores_i.squeeze(-1)[mask_i])
            ci += self.t_sections[i]

        return out

    @torch.no_grad()
    def get_posterior(self, transitions, type_mask, labels):
        """
        H(rule) = (1/N\phi(q) + \phi(q, rule)\phi(rule))y*/2 + \log AutoReg(rule|q)
        params:
            transitions: [B, N, N, 2R+1]
            type_mask: [B, N, N, R]
            labels: [B, N, N, R]  +-1
        """
        B, N, _, R = type_mask.size()
        prior = self.rule_scores.split(self.c_sections.cpu().numpy().tolist(), dim=0)  # List[[Nc_i], ...]
        counts = self.rule_counts.split(self.c_sections.cpu().numpy().tolist(), dim=0)  # List[[Nc_i], ...]

        scores = self.path_scorer(transitions, self.rules).exp()  # [B, N, N, Nc]
        scores_split = scores.split(self.c_sections.cpu().numpy().tolist(), dim=-1)  # List[[B, N, N, Nc_i], ...]
        posterior_counter = [Counter() for _ in self.triples]

        ci = 0
        for i in range(R):
            if self.t_sections[i] == 0:
                continue
            for ti in range(self.t_sections[i]):
                scores_i = (scores_split[ci + ti].unsqueeze(-1) * self.weights[ci + ti][None, None, None, ...]).squeeze(-1) / counts[ci + ti][None, None, None, ...] \
                    + self.biases[ci + ti] / self.Ns  # [B, N, N, Nc]
                H = scores_i * labels[:, :, :, i][..., None] * 0.5 + prior[ci + ti][None, None, None, ...]
                mask_i = (type_mask[:, :, :, i] == ti)  # [B, N, N]
                H = H[mask_i]  # [Ni, Nc]

                H_exp = H.exp() * counts[ci + ti][None, ...]
                probs = H_exp / H_exp.sum(dim=-1).unsqueeze(-1)
                posterior_samples = Categorical(probs).sample()  # [Ni]

                for sample in posterior_samples:
                    tidx = ci + ti
                    cidx = sample + int(sum(self.c_sections[:tidx]))
                    rule = tuple(self.rules[cidx].cpu().numpy().tolist())
                    posterior_counter[tidx][rule] += 1
            ci += self.t_sections[i]
        
        return posterior_counter


class PathScorer(nn.Module):
    def __init__(self):
        super(PathScorer, self).__init__()

    def forward(self, transitions, chains):
        """
        calculate path scores with dynamic programming
        params:
            transitions: [B, N, N, 2R+1] transition score (logits)
            bodies: [Nc, L]
            chains
        """
        Nc, L = chains.size()

        scores = torch.max(
            transitions.index_select(-1, chains[:, 0]).unsqueeze(-2) + \
            transitions.index_select(-1, chains[:, 1]).unsqueeze(-4), dim=2
        ).values  # [B, N, N, Nc]

        for i in range(2, L):
            scores = torch.max(
                scores.unsqueeze(-2) + \
                transitions.index_select(-1, chains[:, i]).unsqueeze(-4),
                dim=2
            ).values

        return scores


if __name__ == "__main__":
    logire = LogiRE('logire-dwie-2')
    logire.EM_optimization()
