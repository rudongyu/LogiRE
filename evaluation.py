import torch
import json
from reader import FeatureReader


class Evaluator():
    def __init__(self, args) -> None:
        self.rules = json.load(open(args.rule_path))
        self.reset()
    # def __init__(self):
    #     self.rules = json.load(open('data/dwie.grules.json'))
    #     self.reset()

    def reset(self):
        self.scores = []
        self.labels = []
        self.intrain = []
        self.rule_correct = torch.zeros(len(self.rules))
        self.rule_total = torch.zeros(len(self.rules))

    def add_item(self, scores, labels, intrain):
        self.scores.append(scores)
        self.labels.append(labels)
        self.intrain.append(intrain)

    def get_ret(self, theta=None):
        scores = torch.cat([scores.reshape(-1) for scores in self.scores])
        labels = torch.cat([labels.reshape(-1) for labels in self.labels]).bool()
        intrain = torch.cat([intrain.reshape(-1) for intrain in self.intrain])

        if theta is None:
            sorted_scores, sorted_indices = scores.sort(descending=True)
            sorted_labels = labels[sorted_indices]
            true = sorted_labels.cumsum(dim=0)
            positive = torch.arange(len(true)).to(true) + 1
            prec = true / positive
            rec = true / labels.sum()
            f1s = 2 * prec * rec / (prec + rec)
            f1s[f1s.isnan()] = 0.
            _, maxi = f1s.max(dim=0)
            theta = sorted_scores[maxi]

        predicted = scores > theta
        prec = (predicted & labels).sum() / predicted.sum()
        rec = (predicted & labels).sum() / labels.sum()
        f1 = 2 * prec * rec / (prec + rec)

        ign_rec = (predicted & labels & (~intrain)).sum() / (labels & (~intrain)).sum()
        ignf1 = 2 * prec * ign_rec / (prec + ign_rec)

        for score in self.scores:
            preds = score > theta
            self.check_rules(preds)

        logic = self.rule_correct / self.rule_total
        logic = logic[~logic.isnan()]
        
        ret = {
            'f1': f1,
            'ignf1': ignf1,
            'theta': theta,
            'logic': logic.mean()
        }

        return ret

    def check_rules(self, preds):
        """
            preds: [N, N, R]
        """
        total = []
        correct = []
        R = preds.size(-1)
        for rule in self.rules:
            head, chain = rule
            head_preds = preds[:, :, head]
            chain_preds = preds[:, :, chain[0]%R]
            if chain[0] >= R:
                chain_preds = chain_preds.transpose(-1, -2)
            for idx in range(1, len(chain)):
                atom_preds = preds[:, :, chain[idx]%R]
                if chain[idx] >= R:
                    atom_preds = atom_preds.transpose(-1, -2)
                chain_preds = (chain_preds.unsqueeze(-1) & atom_preds.unsqueeze(0)).sum(1) > 0
            total.append(chain_preds.sum())
            correct.append((chain_preds & head_preds).sum())

        self.rule_correct += torch.stack(correct, dim=0)
        self.rule_total += torch.stack(total, dim=0)


if __name__ == '__main__':
    fpath = '../baselines/zhou-etal-2021/save/dwie-bert.features'
    reader = FeatureReader(fpath)
    dev_data = reader.read('dev')
    test_data = reader.read('test')
    evaluator = Evaluator()
    for item in dev_data:
        evaluator.add_item(item['logits'].sigmoid(), item['labels'], item['in_train'])
    dev_ret = evaluator.get_ret()
    theta = dev_ret['theta']

    evaluator.reset()
    for item in test_data:
        evaluator.add_item(item['logits'].sigmoid(), item['labels'], item['in_train'])
    test_ret = evaluator.get_ret(theta)

    print(dev_ret)
    print(test_ret)
