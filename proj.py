import random
import numpy as np
import matplotlib.pyplot as plt

class Consumers:
    def __init__(self, K0=100, K1=120):
        self.K0 = K0
        self.K1 = K1
        self.total_services_received = 0
        self.total_payment = 0

    def g(self, u):
        if self.K1 == float('inf'):
            return u
        else:
            return min(max(u, self.K0), self.K1)

    def receive_services(self, units):
        payment = self.g(units)
        self.total_services_received += units
        self.total_payment += payment
        return payment
    
class SUS:
    def __init__(self, capacity):
        self.capacity = capacity
        self.storage = 0

    def store(self, units):
        to_store = min(units, self.capacity - self.storage)
        self.storage += to_store
        return to_store

    def retrieve(self, units):
        to_retrieve = min(units, self.storage)
        self.storage -= to_retrieve
        return to_retrieve

class Device:
    def __init__(self, id, phase=1, state='UP'):
        self.id = id
        self.phase = phase
        self.state = state
        self.repairing = False
        self.was_provider = False
        self.last_reward = 0
        self.history_rewards = []

    def decision(self, recommendation):
        return recommendation

    def update(self, commit_decision, failure_probs, service_costs, repair_costs, F):
        reward = 0
        self.was_provider = False
        if self.state == 'UP':
            if commit_decision == 'COMMIT':
                if random.random() < failure_probs[self.phase-1]:
                    self.state = 'DOWN'
                    self.repairing = True
                    reward -= repair_costs[self.phase-1]
                    self.phase = 1
                else:
                    reward -= service_costs[self.phase-1]
                    self.phase += 1
                    if self.phase >= F:
                        self.phase = F-1
                    self.was_provider = True
            else:
                self.state = 'DOWN'
                self.repairing = False
                self.phase = 1
        elif self.state == 'DOWN':
            self.state = 'UP'
            self.phase = 1
        self.last_reward = reward
        self.history_rewards.append(reward)
        return reward


def rule_motywacyjna(dev, sph, l_prev, s_prev):
    # Reguła motywacyjna: COMMIT  jeśli zysk netto > 0
    if dev.state != 'UP':
        return 'ABSTAIN'
    F = len(sph.failure_probs)
    if dev.phase >= F: # faza F - nie COMMIT
        return 'ABSTAIN'
    # szacowana płatność: proporcjonalna do h(i)
    p_fail = sph.failure_probs[dev.phase-1]
    h_i = pow(dev.phase, sph.alpha)
    l_i = l_prev[dev.phase-1] if dev.phase-1 < len(l_prev) else 1
    sum_h = sum([pow(j+1, sph.alpha)*l_prev[j] for j in range(len(l_prev))]) + h_i # +h_i: optymistycznie, że sam będzie
    expected_payment = sph.last_payment_to_devices * (h_i / sum_h) if sum_h > 0 else 0
    expected_profit = (1 - p_fail) * (expected_payment - sph.service_costs[dev.phase-1]) - p_fail * sph.repair_costs[dev.phase-1]
    return 'COMMIT' if expected_profit > 0 else 'ABSTAIN'

def rule_naiwna(dev, sph, naiwne_commit_ids):
    # Naiwna: tylko wybrane urządzenia z listy commitują
    if dev.state != 'UP':
        return 'ABSTAIN'
    F = len(sph.failure_probs)
    if dev.phase >= F:
        return 'ABSTAIN'
    return 'COMMIT' if dev.id in naiwne_commit_ids else 'ABSTAIN'

def rule_adaptive_threshold(dev, sph, threshold):
    # Klasyczna adaptacja progu decyzyjnego
    if dev.state != 'UP':
        return 'ABSTAIN'
    F = len(sph.failure_probs)
    if dev.phase >= F:
        return 'ABSTAIN'
    p_fail = sph.failure_probs[dev.phase-1]
    h_i = pow(dev.phase, sph.alpha)
    # zakładamy, że wszystkich w tej fazie jest ~średnia liczba
    sum_h = h_i * 5
    expected_payment = sph.last_payment_to_devices * (h_i / sum_h) if sum_h > 0 else 0
    expected_profit = (1 - p_fail) * (expected_payment - sph.service_costs[dev.phase-1]) - p_fail * sph.repair_costs[dev.phase-1]
    return 'COMMIT' if expected_profit > threshold else 'ABSTAIN'

class SPH:
    def __init__(self, nU, nSUS, failure_probs, service_costs, repair_costs, K0=100, K1=120, alpha=0, F=5, rule='motywacyjna', naiwne_commit_ids=None, adapt_threshold=0):
        self.devices = [Device(id=i) for i in range(nU)]
        self.sus = SUS(nSUS)
        self.consumers = Consumers(K0, K1)
        self.failure_probs = failure_probs
        self.service_costs = service_costs
        self.repair_costs = repair_costs
        self.K0 = K0
        self.K1 = K1
        self.alpha = alpha
        self.F = F
        self.rule = rule
        self.last_l = [0]*F
        self.last_s = 0
        self.last_payment_to_devices = 1
        self.naiwne_commit_ids = naiwne_commit_ids or []
        self.adapt_threshold = adapt_threshold

    def recommend(self, device):
        if self.rule == 'motywacyjna':
            return rule_motywacyjna(device, self, self.last_l, self.last_s)
        elif self.rule == 'naiwna':
            return rule_naiwna(device, self, self.naiwne_commit_ids)
        elif self.rule == 'adaptive':
            return rule_adaptive_threshold(device, self, self.adapt_threshold)
        return 'ABSTAIN'

    def optimize_x(self, u, s):
        # Zgodnie z wytycznymi: x = z - y, -s <= x <= min(nSUS - s, u)
        x_min = max(-s, -u)
        x_max = min(self.sus.capacity - s, u)
        best_x = 0
        best_value = -float('inf')
        for x in range(x_min, x_max+1):
            val = self.consumers.g(u-x) + x
            if val > best_value:
                best_value = val
                best_x = x
        return best_x

    def step(self):
        total_services = 0
        rewards = [0] * len(self.devices)
        l = [0]*self.F
        recommendations = [self.recommend(d) for d in self.devices]
        for i, (device, rec) in enumerate(zip(self.devices, recommendations)):
            if device.state == 'UP':
                decision = device.decision(rec)
                reward = device.update(decision, self.failure_probs, self.service_costs, self.repair_costs, self.F)
                if device.was_provider:
                    total_services += 1
                    l[device.phase-1] += 1
                rewards[i] = reward
            else:
                rewards[i] = device.update(None, self.failure_probs, self.service_costs, self.repair_costs, self.F)
        u = total_services
        s = self.sus.storage
        x_opt = self.optimize_x(u, s)
        z, y = (x_opt, 0) if x_opt >= 0 else (0, -x_opt)
        # z*y == 0
        if z > 0:
            self.sus.store(z)
        if y > 0:
            self.sus.retrieve(y)
        services_to_consumer = u - z + y
        payment_from_consumers = self.consumers.receive_services(services_to_consumer)
        payment_to_devices = payment_from_consumers + z - y
        providers = [(i, d) for i, d in enumerate(self.devices) if d.was_provider]
        phases = [d.phase for _, d in providers]
        weights = [pow(p, self.alpha) for p in phases]
        total_weight = sum(weights)
        payments = [payment_to_devices * (w / total_weight) if total_weight > 0 else 0 for w in weights]
        for (i, _), pay in zip(providers, payments):
            rewards[i] += pay
        # Zapamiętanie na następną iterację
        self.last_l = l
        self.last_s = self.sus.storage
        self.last_payment_to_devices = payment_to_devices
        return {
            'services_to_consumer': services_to_consumer,
            'payment_to_devices': payment_to_devices,
            'average_device_reward': np.mean(rewards),
            'total_reward': np.sum(rewards),
            'consumer_total_services': self.consumers.total_services_received,
            'consumer_total_payment': self.consumers.total_payment,
            'rewards': rewards,
            'phases': [d.phase for d in self.devices],
            'commit_successes': sum([1 for i, d in enumerate(self.devices) if recommendations[i]=='COMMIT' and d.was_provider]),
            'commit_attempts': sum([1 for i, d in enumerate(self.devices) if recommendations[i]=='COMMIT']),
            'commit_rewards': [d.last_reward for i, d in enumerate(self.devices) if recommendations[i]=='COMMIT'],
            'abstain_virtual_rewards': [self._virtual_commit_reward(d) for i, d in enumerate(self.devices) if recommendations[i]=='ABSTAIN' and d.state=='UP'],
        }

    def _virtual_commit_reward(self, device):
        if device.state != 'UP':
            return 0
        phase = device.phase
        F = self.F
        if phase >= F:
            return 0
        p_fail = self.failure_probs[phase-1]
        h_i = pow(phase, self.alpha)
        sum_h = h_i * 5
        expected_payment = self.last_payment_to_devices * (h_i / sum_h) if sum_h > 0 else 0
        expected_profit = (1 - p_fail) * (expected_payment - self.service_costs[phase-1]) - p_fail * self.repair_costs[phase-1]
        return expected_profit

def simulate(T=1000, nU=150, nSUS=20, K0=100, K1=120, alpha=1, F=5, zeta=0.5):
    failure_probs = [0.1, 0.2, 0.3, 0.4, 1.0]
    service_costs = [0.5] * 5
    repair_costs = [0.5, 0.5, 0.7, 1.5, 3]
    # Motywacyjna reguła
    sph_mot = SPH(nU, nSUS, failure_probs, service_costs, repair_costs, K0, K1, alpha, F, rule='motywacyjna')
    # Naiwna reguła: co cykl inny zestaw urządzeń UP w fazach 1..F-1
    sph_naiwna = SPH(nU, nSUS, failure_probs, service_costs, repair_costs, K0, K1, alpha, F, rule='naiwna')
    # Adaptacyjny próg (na początek 0.1, potem aktualizacja)
    sph_adapt = SPH(nU, nSUS, failure_probs, service_costs, repair_costs, K0, K1, alpha, F, rule='adaptive', adapt_threshold=0.1)
    # Statystyki
    stats = {
        'motywacyjna': {'services': [], 'payments': [], 'avg_reward': [], 'total_reward': [], 'cum_valuation': [],
                        'commit_success_rate': [], 'avg_commit_reward': [], 'avg_abstain_virtual': []},
        'naiwna': {'services': [], 'payments': [], 'avg_reward': [], 'total_reward': [], 'cum_valuation': [],
                   'commit_success_rate': [], 'avg_commit_reward': [], 'avg_abstain_virtual': []},
        'adaptive': {'services': [], 'payments': [], 'avg_reward': [], 'total_reward': [], 'cum_valuation': [],
                     'commit_success_rate': [], 'avg_commit_reward': [], 'avg_abstain_virtual': []},
    }
    for t in range(T):
        # Naiwna: losujemy zeta * urządzeń UP w fazach < F
        up_ids = [d.id for d in sph_naiwna.devices if d.state == 'UP' and d.phase < F]
        n_commit = int(zeta * len(up_ids))
        naiwne_commit_ids = set(random.sample(up_ids, n_commit)) if n_commit > 0 else set()
        sph_naiwna.naiwne_commit_ids = naiwne_commit_ids
        # Motywacyjna
        res_mot = sph_mot.step()
        res_naiw = sph_naiwna.step()
        res_adap = sph_adapt.step()
        # Update adaptive threshold (klasyczna adaptacja progu)
        last_commit_rewards = res_adap['commit_rewards']
        if last_commit_rewards:
            sph_adapt.adapt_threshold = 0.9 * sph_adapt.adapt_threshold + 0.1 * np.mean(last_commit_rewards)
        # Statystyki
        for name, res in zip(['motywacyjna', 'naiwna', 'adaptive'], [res_mot, res_naiw, res_adap]):
            stats[name]['services'].append(res['services_to_consumer'])
            stats[name]['payments'].append(res['payment_to_devices'])
            stats[name]['avg_reward'].append(res['average_device_reward'])
            stats[name]['total_reward'].append(res['total_reward'])
            if stats[name]['cum_valuation']:
                stats[name]['cum_valuation'].append(stats[name]['cum_valuation'][-1] + res['services_to_consumer'])
            else:
                stats[name]['cum_valuation'].append(res['services_to_consumer'])
            # commit success rate
            cs = res['commit_successes']
            ca = res['commit_attempts']
            stats[name]['commit_success_rate'].append(cs / ca if ca > 0 else 0)
            # avg commit reward
            stats[name]['avg_commit_reward'].append(np.mean(res['commit_rewards']) if res['commit_rewards'] else 0)
            # avg abstain virtual
            stats[name]['avg_abstain_virtual'].append(np.mean(res['abstain_virtual_rewards']) if res['abstain_virtual_rewards'] else 0)
    return stats

def plot_results(stats, T, zeta, cust, count):
    x = np.arange(T)
    plt.figure(figsize=(15, 12))
    plt.subplot(231)
    plt.title("Chwilowa waluacja usług (do konsumentów)")
    for k in stats:
        plt.plot(x, stats[k]['services'], label=k)
    plt.subplot(232)
    plt.title("Skumulowana waluacja usług (do konsumentów)")
    for k in stats:
        plt.plot(x, stats[k]['cum_valuation'], label=k)
    plt.legend()
    plt.subplot(233)
    plt.title("Średni zysk netto urządzenia")
    for k in stats:
        plt.plot(x, stats[k]['avg_reward'], label=k)
    plt.subplot(234)
    plt.title("Wskaźnik nieprzerwanego dostarczania")
    for k in stats:
        plt.plot(x, stats[k]['commit_success_rate'], label=k)
    plt.subplot(235)
    plt.title("Śr. zysk netto urządzeń (COMMIT)")
    for k in stats:
        plt.plot(x, stats[k]['avg_commit_reward'], label=k)
    plt.subplot(236)
    plt.title("Wirtualny (no-regret) zysk ABSTAIN")
    plt.suptitle(f"symulacja systemu [NSUS: {cust} NU: {zeta}]")
    for k in stats:
        plt.plot(x, stats[k]['avg_abstain_virtual'], label=k)
    plt.tight_layout()
    plt.savefig(f"figury/{count}.png")
    plt.close()

if __name__ == "__main__":
    T = 500
    nU = 50
    nSUS = 10
    K0 = 10
    K1 = 15
    alpha = 1
    F = 5
    zeta = 0.5  # udział commitów w naiwnej regule
    
    count = 1
    
    for nsus in np.arange(10, 100, 10):
        for nu in np.arange(20, 200, 10):
            stats = simulate(T=T, nU=nu, nSUS=nsus, K0=K0, K1=K1, alpha=alpha, F=F, zeta=zeta)
            plot_results(stats, T, nsus, nu, count)
        
            count += 1