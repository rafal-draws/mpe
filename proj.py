import random
import numpy as np
import matplotlib.pyplot as plt

# podpunkt 1


class Consumers:
    def __init__(self, K0=100, K1=120):
        self.K0 = K0
        self.K1 = K1
        self.total_services_received = 0
        self.total_payment = 0

    # podpunkt 14 - ograniczona zdolnosc konsumpcji

    def g(self, u):
        if self.K1 == float('inf'):
            return u
        else:
            return min(max(u, self.K0), self.K1)

    # podpunkt 3
    def receive_services(self, units):
        payment = self.g(units)
        self.total_services_received += units
        self.total_payment += payment
        return payment

# =============================
# Bufor SUS
# =============================


class SUS:
    def __init__(self, capacity):
        self.capacity = capacity
        self.storage = 0

    # podpunkt 6
    def store(self, units):
        to_store = min(units, self.capacity - self.storage)
        self.storage += to_store
        return to_store

    # podpunkt 6
    def retrieve(self, units):
        to_retrieve = min(units, self.storage)
        self.storage -= to_retrieve
        return to_retrieve

# podpunkt 1


class Device:

    # podpunkt 12 - start faza 1 i stan UP (bez losowej inicjalizacji)
    def __init__(self, id, phase=1, state='UP'):
        self.id = id
        self.phase = phase

        # podpunkt 9 -> tryby pracy urzadze up, down i logika commit abstrain
        self.state = state
        self.repairing = False
        self.was_provider = False

    def decision(self, recommendation):
        return recommendation

    # podpunkt 2
    def update(self, commit_decision, failure_probs, service_costs, repair_costs):
        reward = 0
        self.was_provider = False

        # podpunkt 9 -> tryby pracy urzadze up, down i logika commit abstrain
        if self.state == 'UP':
            if commit_decision == 'COMMIT':

                # podpunkt 8 - moze wywolac awarie przy commit
                # podpunkt 10 obslucha przejsc faz i napraw state = down reset phase 0; sukces phase += 1
                if random.random() < failure_probs[self.phase-1]:
                    self.state = 'DOWN'
                    self.repairing = True
                    reward -= repair_costs[self.phase-1]
                else:
                    reward -= service_costs[self.phase-1]
                    self.phase += 1
                    if self.phase > len(failure_probs):
                        self.phase = len(failure_probs)
                    self.was_provider = True
            else:
                self.state = 'DOWN'
                self.repairing = False
        elif self.state == 'DOWN':
            self.state = 'UP'
            self.phase = 1
        return reward

# =============================
# Modul SPH
# =============================


# podpunkt 4
class SPH:
    def __init__(self, nU, nSUS, failure_probs, service_costs, repair_costs, K0=100, K1=120, alpha=0):

        # podpunkt 5 nU urzadzen
        self.devices = [Device(id=i) for i in range(nU)]
        # podpunkt 5 SUS
        self.sus = SUS(nSUS)
        self.consumers = Consumers(K0, K1)
        self.failure_probs = failure_probs
        self.service_costs = service_costs
        self.repair_costs = repair_costs
        self.K0 = K0
        self.K1 = K1
        self.alpha = alpha

    # podpunkt 15

    def recommend(self, device):
        if device.state == 'UP' and device.phase < len(self.failure_probs):
            p_fail = self.failure_probs[device.phase-1]
            expected_profit = (
                1 - p_fail) * (1 - self.service_costs[device.phase-1]) - p_fail * self.repair_costs[device.phase-1]
            return 'COMMIT' if expected_profit > 0 else 'ABSTAIN'
        return 'ABSTAIN'

    def optimize_x(self, u, s):
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
        recommendations = [self.recommend(d) for d in self.devices]

        for i, (device, rec) in enumerate(zip(self.devices, recommendations)):

            # podpunkt 2
            if device.state == 'UP':
                decision = device.decision(rec)
                reward = device.update(
                    decision, self.failure_probs, self.service_costs, self.repair_costs)
                if device.was_provider:
                    total_services += 1
                rewards[i] = reward
            else:
                rewards[i] = device.update(
                    None, self.failure_probs, self.service_costs, self.repair_costs)

        # podpunkt 13 logika SPH pod koniec cyklu SPH-STP SPH-PDP
        u = total_services
        s = self.sus.storage
        x_opt = self.optimize_x(u, s)
        z, y = (x_opt, 0) if x_opt >= 0 else (0, -x_opt)

        self.sus.store(z)
        self.sus.retrieve(y)

        services_to_consumer = u - z + y
        payment_from_consumers = self.consumers.receive_services(
            services_to_consumer)
        payment_to_devices = payment_from_consumers + z - y

        # Dystrybucja płatności według h(i) = i^alpha
        providers = [(i, d)
                     for i, d in enumerate(self.devices) if d.was_provider]
        phases = [d.phase for _, d in providers]
        weights = [pow(p, self.alpha) for p in phases]
        total_weight = sum(weights)
        payments = [payment_to_devices *
                    (w / total_weight) if total_weight > 0 else 0 for w in weights]

        for (i, _), pay in zip(providers, payments):
            rewards[i] += pay

        return {
            'services_to_consumer': services_to_consumer,
            'payment_to_devices': payment_to_devices,
            'average_device_reward': np.mean(rewards),
            'total_reward': np.sum(rewards),
            'consumer_total_services': self.consumers.total_services_received,
            'consumer_total_payment': self.consumers.total_payment
        }

# =============================
# Funkcja symulacji
# =============================

# podpunkt 7 - symulacja po T cyklach, kazdy to wywolanie step()


def simulate(T=1000, nU=150, nSUS=20, K0=100, K1=120, alpha=1,

             # podpunkt 11 - model niezawodnosci i kosztow zalezny od faz
             failure_probs=[0.1, 0.2, 0.3, 0.4, 1.0],
             service_costs=[0.7, 0.5, 0.4, 0.3, 0.9],
             repair_costs=[0.7, 0.5, 0.7, 1.5, 3]):

    sph = SPH(nU, nSUS, failure_probs, service_costs,
              repair_costs, K0, K1, alpha)

    results = {
        'services_to_consumer': [],
        'payment_to_devices': [],
        'average_device_reward': [],
        'total_reward': [],
        'consumer_total_services': [],
        'consumer_total_payment': []
    }

    for t in range(T):
        res = sph.step()
        for key in results:
            results[key].append(res[key])

    return results


def plot_results(results):
    T = len(results['services_to_consumer'])
    time = list(range(1, T + 1))

    fig, axs = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Wyniki symulacji systemu usługowego (SPH)', fontsize=18)

    axs[0, 0].plot(time, results['services_to_consumer'],
                   label='Usługi dla konsumentów', color='green')
    axs[0, 0].set_title('Usługi dla konsumentów')
    axs[0, 0].set_ylabel('Liczba usług')
    axs[0, 0].grid(True)

    axs[0, 1].plot(time, results['payment_to_devices'],
                   label='Wypłaty dla urządzeń', color='blue')
    axs[0, 1].set_title('Wypłaty dla urządzeń')
    axs[0, 1].set_ylabel('Kwota')
    axs[0, 1].grid(True)

    axs[1, 0].plot(time, results['average_device_reward'],
                   label='Średni zysk urządzenia', color='orange')
    axs[1, 0].set_title('Średni zysk urządzenia')
    axs[1, 0].set_ylabel('Zysk')
    axs[1, 0].grid(True)

    axs[1, 1].plot(time, results['total_reward'],
                   label='Całkowity zysk urządzeń', color='purple')
    axs[1, 1].set_title('Całkowity zysk urządzeń')
    axs[1, 1].set_ylabel('Zysk')
    axs[1, 1].grid(True)

    axs[2, 0].plot(time, results['consumer_total_services'],
                   label='Skumulowane usługi', color='darkgreen')
    axs[2, 0].set_title('Skumulowane usługi dla konsumentów')
    axs[2, 0].set_xlabel('Cykl')
    axs[2, 0].set_ylabel('Usługi')
    axs[2, 0].grid(True)

    axs[2, 1].plot(time, results['consumer_total_payment'],
                   label='Skumulowane płatności', color='darkred')
    axs[2, 1].set_title('Skumulowane płatności od konsumentów')
    axs[2, 1].set_xlabel('Cykl')
    axs[2, 1].set_ylabel('Płatność')
    axs[2, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_combined(results):
    T = len(results['services_to_consumer'])
    time = list(range(1, T + 1))

    plt.figure(figsize=(12, 6))
    plt.plot(time, results['services_to_consumer'],
             label='Usługi dla konsumentów', color='green')
    plt.plot(time, results['average_device_reward'],
             label='Średni zysk urządzenia', color='orange')
    plt.plot(time, results['total_reward'],
             label='Całkowity zysk urządzeń', color='purple')

    plt.title('Usługi i zyski urządzeń w czasie')
    plt.xlabel('Cykl')
    plt.ylabel('Wartość')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    """
    T - ilosc cykli
    nU - liczba urzadzen
    nSUS - pojemnosc bufora

    funkcja waluacji konsumentow g(u)
    k0 - (def 100)
    k1 - 120 lub 'inf'
    alpha - funkcja promocji wyzszej fazy

    """
    result = simulate(T=500, nU=500, nSUS=100, K0=100, K1=120, alpha=0)

    for key in result.keys():
        print(f"{key}: {result[key][-1]}")
    plot_results(result)
    plot_combined(result)
