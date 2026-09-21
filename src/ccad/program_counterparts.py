import torch


class SourceActionBank:
    def __init__(self, parts):
        self.parts = parts
        self.statistics = {}

    def observe(self, site, part, action):
        if part not in self.parts:
            return
        matrix = action.reshape(-1, action.shape[-1]).double()
        key = site+'__'+part
        if key not in self.statistics:
            self.statistics[key] = torch.zeros(matrix.shape[1], matrix.shape[1], dtype=torch.float64, device=matrix.device)
        energy = matrix.square().sum()
        if float(energy) > 1e-20:
            self.statistics[key] += matrix.T @ matrix / energy

    def arrays(self):
        return {key:value.cpu().numpy() for key,value in self.statistics.items()}


def shared_support_order(decoder, covariance, count):
    values, vectors = torch.linalg.eigh(covariance.double())
    chosen = values > values[-1].clamp_min(1e-20) * 1e-10
    if not bool(chosen.any()):
        raise ValueError('No signal for shared-support selection')
    signals = (vectors[:, chosen] * values[chosen].sqrt()).float()
    atoms = decoder.float() / decoder.float().norm(dim=1, keepdim=True).clamp_min(1e-12)
    correlations = atoms @ signals
    basis = []
    selected = []
    for _ in range(count):
        scores = correlations.square().sum(-1)
        if selected:
            scores[selected] = -torch.inf
        index = int(scores.argmax())
        direction = atoms[index].clone()
        if basis:
            q = torch.stack(basis, dim=1)
            # 重复投影控制浮点正交误差。
            direction -= q @ (q.T @ direction)
            direction -= q @ (q.T @ direction)
        length = direction.norm()
        if float(length) < 1e-6:
            raise ValueError('Selected decoder directions are linearly dependent')
        direction /= length
        correlations -= (atoms @ direction).unsqueeze(1) * (direction @ signals).unsqueeze(0)
        basis.append(direction)
        selected.append(index)
    return selected


class ProgramCounterparts:
    def __init__(self, dictionaries, parts):
        self.dictionaries = dictionaries
        self.parts = parts
        self.statistics = {}

    def observe(self, site, part, desired, action, coefficients, indices):
        if part not in self.parts:
            return
        decoder = self.dictionaries[site].decoder.weight.T
        key = (site, part)
        if key not in self.statistics:
            self.statistics[key] = dict(
                state=torch.zeros(decoder.shape[1], decoder.shape[1], dtype=torch.float64, device=decoder.device),
                action=torch.zeros(decoder.shape[1], decoder.shape[1], dtype=torch.float64, device=decoder.device),
                participation=torch.zeros(len(decoder), dtype=torch.float64, device=decoder.device),
                documents=0)
        stats = self.statistics[key]
        for name, values in [('state', desired), ('action', action)]:
            matrix = values.reshape(-1, values.shape[-1]).double()
            energy = matrix.square().sum()
            if float(energy) > 1e-20:
                stats[name] += matrix.T @ matrix / energy
        masses = coefficients.double().square() * decoder.norm(dim=1).double()[indices].square()
        energy = masses.sum()
        if float(energy) > 1e-20:
            stats['participation'].scatter_add_(0, indices.flatten(), (masses/energy).flatten())
        stats['documents'] += 1

    def export(self, sizes):
        result = {f'{method}_{size}': {} for method in ['participation', 'action_span', 'state_span'] for size in sizes}
        arrays = {}
        for (site, part), stats in self.statistics.items():
            decoder = self.dictionaries[site].decoder.weight.T
            order = torch.argsort(stats['participation'], descending=True, stable=True)
            orders = {'participation': order[stats['participation'][order] > 0].tolist()}
            for name in ['action', 'state']:
                if float(stats[name].trace()) > 1e-20:
                    orders[name+'_span'] = shared_support_order(decoder, stats[name], max(sizes))
                else:
                    # 此位置该部分没有局部源动作，其目标传播仍可能需要成员。
                    orders[name+'_span'] = []
            for method, order in orders.items():
                for size in sizes:
                    result[f'{method}_{size}'].setdefault(part, {})[site] = order[:size]
            for name in ['state', 'action', 'participation']:
                arrays[site+'__'+part+'__'+name] = stats[name].cpu().numpy()
        return result, arrays
