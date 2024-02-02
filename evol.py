import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import solve_ivp
import webbrowser
import math
import networkx as nx
from datetime import datetime
import os

seed_time = datetime.now()
seed = seed_time.strftime("%Y%m%d_%H%M%S")
rng = np.random.default_rng(seed=int(seed))
folder_name = f"/home/jakab/Goldford_Results/Invade/{seed}"

if not os.path.exists(folder_name):
    os.mkdir(folder_name)

def matrix_cell_html(value):
    if value != 0:
        # Non-zero values get a colored background
        return f'<td style="background-color: #FFD700;">{value:.2f}</td>'
    else:
        # Zero values remain without background color
        return f'<td>{value:.2f}</td>'

class CustomError(Exception):
    pass


class Object(object):
    pass


class GCRM:
    def __init__(self, n_species, n_resources, spec, spec_var, n_families, inflow_rate, results_path, dcase=None,
                 rcase=None, mcase=None, cutoffvalue=float(0), dirichlet_hyper=100, between_family_var=0.1,
                 inside_family_var=0.05, m_val=float(1), invade=False, n_invaders=int(0), t_invaders=int(0),
                 first_invasion=int(0), h=float(1)):
        self.results_path = results_path
        self.n_species = n_species
        self.n_resources = n_resources
        self.n_families = n_families
        self.dcase = dcase
        self.mcase = mcase
        self.rcase = rcase
        self.cutoffvalue = cutoffvalue
        self.dirichlet_hyper = dirichlet_hyper
        self.between_family_var = between_family_var
        self.inside_family_var = inside_family_var
        self.h = h
        self.invade = invade
        if invade:
            if not (n_invaders and t_invaders and first_invasion):
                raise CustomError("Invaders are present, please set the value of n_invaders, t_invaders and first_invasion")
            self.t_invaders = t_invaders
            self.n_invaders = n_invaders
            self.first_invasion = first_invasion

        self.inflow_rate = inflow_rate

        match dcase:
            case "random_refined":
                self.d_matrix_html = "<p>D matrix is refined but random</p>"
                if n_resources < 3:
                    raise CustomError("n_resources must be higher than 3 when dcase:random_refined is used")
                n_levels = 1 + np.sqrt(n_resources)
                levels = np.arange(0, n_levels)
                remaining = n_resources - len(levels)
                random_choices = rng.choice(levels[1:-1], size=remaining)
                levels = np.sort(np.concatenate((levels, random_choices)))
                energy_surplus = 1
                self.W = np.ones(n_resources)
                for b in range(1, n_resources):
                    self.W[b] = energy_surplus * levels[b] ** 3 + 2 ** ((n_levels - 1) - levels[b])

                self.W_ba = np.zeros((n_resources, n_resources))
                self.D = np.zeros((n_resources, n_resources))
                for i in range(n_resources):
                    for j in range(n_resources):
                        if levels[i] - levels[j] > 0:
                            self.D[i, j] = 2 ** (levels[i] - levels[j])
                            self.W_ba[i, j] = self.W[j] - self.D[i, j] * self.W[i]

                print("Overwrite W")
            case "nullmodel":
                self.d_matrix_html = "<p>D matrix is refined and has fixed values</p>"
                if n_resources != 10:
                    raise CustomError("n_resources must be 10 when dcase:saci is used")
                levels = np.array([0, 1, 1, 2, 2, 2, 2, 3, 3, 4])

                self.W = np.array([15, 7, 7, 3, 3, 3, 3, 1, 1, 0])
                self.W_ba = np.zeros((n_resources, n_resources))
                self.D = np.zeros((n_resources, n_resources))
                for i in range(n_resources):
                    for j in range(n_resources):
                        if levels[i] - levels[j] > 0:
                            self.D[i, j] = 2 ** (levels[i] - levels[j])
                            print("i", self.W[i])
                            print("j", self.W[j])
                            self.W_ba[i, j] = self.W[j] - self.D[i, j] * self.W[i]
                print(self.D)
                print(self.W_ba)

                print("Overwrite W")

            case "lt":
                self.d_matrix_html = "<p>D is a random lower triangular matrix</p>"
                self.D = np.tril(rng.uniform(0, 1, n_resources * n_resources).reshape(n_resources, n_resources), k=-1)
                print("D is a lower triangular matrix")
            case _:
                self.d_matrix_html = "<p>D is a random matrix</p>"
                self.D = rng.uniform(0, 1, n_resources * n_resources).reshape(n_resources,
                                                                              n_resources) * 1 / self.n_resources
                print("D is a random matrix")

        match rcase:
            case "rand1":
                self.primary_idx = rng.integers(0, n_resources, 1)[0]
                print("1 primary resource is picked randomly")
            case _:
                self.primary_idx = 0
                print("The first resource is the primary resource")

        self.priors = []
        react = [int(self.n_resources/2), int(self.n_resources/5)]
        self.n_reactions_fam = rng.choice(react, self.n_families, replace=True)
        for i in range(n_families):
            D_copy = self.D.copy()
            C_i = np.zeros((n_resources, n_resources))
            rand_consumption_floats = rng.uniform(0, 1, self.n_reactions_fam[i])
            rescaled = np.array([value / sum(rand_consumption_floats) * self.dirichlet_hyper for value in
                        rand_consumption_floats])
            rescaled = rescaled
            for k in range(self.n_reactions_fam[i]):
                true_indices = np.array(np.where(D_copy)).T
                chosen_indices = true_indices[rng.choice(true_indices.shape[0])]
                C_i[chosen_indices[0], chosen_indices[1]] = rescaled[k]

                ### PREMISE: no molecule should be used as a product and as a reactant at the same time, by the same species
                D_copy[:, chosen_indices[0]] = np.zeros(n_resources)  # The product cannot be used as a reactant
                D_copy[chosen_indices[0], chosen_indices[1]] = False  # Ensure that reaction is not overwritten
                D_copy[chosen_indices[1], :] = np.zeros(n_resources)  # Reactant cannot be produced

            self.priors.append(C_i)

        self.metacom = []
        self.family_ids = np.array([])
        self.all_m = np.array([])
        self.m_priors = rng.normal(m_val, self.between_family_var, n_families)
        for idx, prior in enumerate(self.priors):
            self.family_ids = np.concatenate((self.family_ids, np.repeat(idx, 100)))
            self.all_m = np.concatenate((self.all_m, rng.normal(1, self.inside_family_var, 100)))
            prior_values = prior[np.where(prior)]
            prior_indices = np.where(prior)
            new_values = rng.dirichlet(alpha=prior_values, size=100)

            for s in range(100):
                sum_powers = np.sum(np.power(new_values[s], self.h))
                new_values[s] = new_values[s] / np.power(sum_powers, 1 / self.h)
                C = np.zeros((n_resources, n_resources))
                C[prior_indices] = new_values[s]
                self.metacom.append(C)

        self.alpha = np.zeros(n_resources)
        self.alpha[self.primary_idx] = 100000
        self.mu = np.ones(n_species)  # ???
        self.tau = np.ones(n_resources)  # ???

    class LocalCom:
        def __init__(self, n_species, n_resources, n_players, indices, com_idx, mapping, invade, n_invaders, populated=True):
            self.species = np.zeros(n_players)
            self.resources = np.zeros(n_resources)
            if populated:
                rands = rng.uniform(0, 1, n_species)
                for rand_idx, spec_idx in enumerate(indices[com_idx]):
                    self.species[mapping.get(spec_idx)] = rands[rand_idx]
                self.resources = rng.uniform(0, 1, n_resources)
            if invade:
                self.species[-n_invaders:] = 0


    def initialise_community(self, repeat):
        if not repeat:
            self.players = rng.choice(len(self.metacom), self.n_species, replace=False)
            self.indices = [self.players.copy()]

            if self.invade:
                remaining = [x for x in list(range(len(self.metacom))) if x not in self.players]
                self.invaders = rng.choice(remaining, self.n_invaders, replace=False)
                self.players = np.concatenate((self.players, self.invaders))
                self.invader_bool = np.concatenate((np.zeros(len(self.players)), np.ones(self.n_invaders)))

            self.n_players = len(self.players)
            self.C = [self.metacom[i] for i in self.players]
            self.mu = np.ones(self.n_players)
            self.n_reactions = np.array([self.n_reactions_fam[int(x)] for x in self.family_ids[self.players]])
            self.n_splits = []
            for i in range(self.n_players):
                val = np.multiply(self.D-1, self.C[i]).sum()
                self.n_splits.append(val)
            self.n_splits = np.array(self.n_splits)

        else:
            perm = rng.permutation(self.n_players)
            self.players = self.players[perm]
            self.n_reactions = self.n_reactions[perm]
            self.n_splits = self.n_splits[perm]
            self.C = [self.C[i] for i in perm]
            self.mu = self.mu[perm]

            for i in range(self.n_players):
                if np.multiply(self.D-1, self.C[i]).sum() != self.n_splits[i]:
                    print("Splits not equal!")
                if np.sum(np.where(self.C[i])[0]) != self.n_reactions[i]:
                    print("Reactions not equal!")


            #self.n_reactions = self.n_reactions[perm]
            #self.n_splits = self.n_splits[perm]

            self.indices = [self.players[:self.n_species].copy()]

        mapping = {original_idx: new_idx for new_idx, original_idx in enumerate(self.players)}
        self.present_species_log = np.arange(0, self.n_players)
        if self.invade:
            self.invaded = np.empty(self.n_players, dtype=object)
            self.invaded[:self.n_species] = np.full(self.n_species, "Yes") #Present from start

        match self.mcase:
            case "ones":
                self.m = np.ones(self.n_players)
                print("Species maintenance values are 1")
            case _:
                self.m = self.all_m[self.players]
                print(f"Species maintenance values are random.  "
                      f"Variance between families: {self.between_family_var}  "
                      f"Variance inside families: {self.inside_family_var}")

        # no need to have a list
        self.communities = []
        self.communities.append(
            GCRM.LocalCom(self.n_species, self.n_resources, self.n_players, self.indices, 0, mapping, invade=self.invade, n_invaders=self.n_invaders))

    def cutoff(self, abundance):
        return np.where(abundance < self.cutoffvalue, 0, abundance)

    def time_development(self, t, y):
        species = y[:self.n_players].copy()
        resources = y[self.n_players:].copy()

        fi = 0.05
        eta = 0.01

        flow_s = -species * self.inflow_rate
        flow_r = -resources * self.inflow_rate

        all_grm = np.zeros(self.n_players)
        all_consumption = np.zeros(self.n_resources)
        all_production = np.zeros(self.n_resources)
        helper_mat = np.zeros((self.n_resources, self.n_resources))
        for i in self.present_species[0]:
            growth_rate_multiplier = (np.multiply(self.C[i] * self.W_ba, resources).sum() - self.m[i] + fi * self.n_reactions[i] + eta * self.n_splits[i])
            consumption = np.multiply(species[i] * self.C[i], resources).sum(axis=0)
            production = np.multiply(self.D * self.C[i] * species[i], resources).sum(axis=1)
            all_grm[i] = growth_rate_multiplier * species[i]
            all_consumption += consumption
            all_production += production
            helper_mat[np.where(self.C[i])] = 1

        self.syntroph_potential.append(all_production[:-1].sum())
        self.realised_syntrophy.append(all_production[:-1].sum() / all_consumption[1:-1].sum())
        self.MCI.append(helper_mat.sum())


        # Calculate final products for timestep
        species_next = all_grm + flow_s

        resources_next = (self.alpha - resources) / self.tau - all_consumption + all_production + flow_r

        if not np.all(np.isfinite(species_next)):
            print("Error: Non-finite values found in initial conditions y0[species]:", species_next)

        if not np.all(np.isfinite(resources_next)):
            print("Error: Non-finite values found in initial conditions y0[resources]:", resources_next)

        return np.concatenate((species_next, resources_next))

    def event(self, t, y):
        return 0 < min(y[y != 0][0:self.n_players] - self.cutoffvalue) or (t - self.last_event_time) < 0.2

    event.terminal = True  # Stops the integration
    event.direction = -1

    def near_equilibrium(self, t, y):
        dydt = self.time_development(t, y)
        threshold = 1e-6
        norm_dydt = np.linalg.norm(dydt) # Resources included
        return threshold - norm_dydt

    near_equilibrium.terminal = True
    near_equilibrium.direction = 0  # Any 0-crossing

    def prepare_1D_array(self, repeat, set_y0):
        if not set_y0.any():
            self.initialise_community(repeat=repeat)
            res0 = self.communities[0].resources
            spec0 = self.communities[0].species
            y0 = np.concatenate((spec0, res0))
        else:
            y0 = set_y0
        if not np.all(np.isfinite(y0)):
            print("Error: Non-finite values found in initial conditions y0:", y0)

        self.present_species = np.where(y0[:self.n_players])

        return y0

    def evol_ivp(self, quant_mut=0.1, type_mut=0.1, ensemble_mut=0.001, t_end=1000):
        y0 = self.prepare_1D_array(repeat=False, set_y0=np.array(False))
        t_start = 0
        t_span = [t_start, t_end]
        all_t = np.array([])
        all_y = np.array([]).reshape(len(y0), 0)
        self.last_event_time = 0

        evol_types = ["quant", "type", "ensemble"]
        prob_sums = quant_mut + type_mut + ensemble_mut
        evol_weights = [quant_mut/prob_sums, type_mut/prob_sums, ensemble_mut/prob_sums]

        while t_span[0] < t_end:
            sol = solve_ivp(self.time_development, t_span, y0, events=[self.event, self.near_equilibrium],
                            method="LSODA",
                            atol=1e-5)  # t_eval=np.linspace(t_start, t_end, num_points), if we want to specify resolution , args=("ivp",)
            if all_t.size == 0:  # If result arrays are empty still, populate them with the first results
                all_t = sol.t
                all_y = sol.y
            else:  # If they are not empty, concatenate the previous and the current results
                all_t = np.concatenate((all_t, sol.t), axis=0)
                all_y = np.concatenate((all_y, sol.y), axis=1)

            # Check if an event was triggered and update the initial condition and t_span accordingly
            if sol.t_events[0].size > 0:
                self.last_event_time = sol.t[-1]
                y0 = np.concatenate(
                    (np.where(sol.y[:self.n_players, -1] < self.cutoffvalue, 0, sol.y[:self.n_players, -1]),
                     sol.y[self.n_players:, -1]))
                self.present_species = np.where(y0[:self.n_players])
                print("t: ", sol.t[-1])
                t_span = [sol.t[-1], t_end]
            elif sol.t_events[1].size > 0:
                selected_outcome = rng.choice(evol_types, p=evol_weights)
                multi = 2
                if selected_outcome == "quant":
                    target = rng.integers(len(self.C))
                    target_C = self.C[target]
                    values = target_C[np.where(target_C)]
                    mutate_idx = rng.integers(0, len(values))
                    values[mutate_idx] = rng.normal(values[mutate_idx], values[mutate_idx]/3, 1)[0]
                    if values[mutate_idx] < 0:
                        values[mutate_idx] = 0
                    resized = self.resize(values, h)
                    target_C[np.where(target_C)] = resized
                    self.C[target] = target_C
                    pass
                elif selected_outcome == "type_loss":
                    target = rng.integers(len(self.C))
                    target_C = self.C[target]
                    values = target_C[np.where(target_C)]
                    mutate_idx = rng.integers(0, len(values))
                    values[mutate_idx] = 0
                    resized = self.resize(values, h)
                    target_C[np.where(target_C)] = resized
                    self.C[target] = target_C
                    pass
                elif selected_outcome == "type_gain":
                    target = rng.integers(len(self.C))
                    target_C = self.C[target]
                    possible_reactions = self.D.copy()
                    for i in range(self.n_resources):
                        if np.any(target_C[i, :]):
                            possible_reactions[:, i] = np.zeros(self.n_resources)
                        if np.any(target_C[:, i]):
                            possible_reactions[i, :] = np.zeros(self.n_resources)
                    pick = rng.choice(np.where(possible_reactions))
                    target_C[pick] = 0.1
                    values = target_C[np.where(target_C)]
                    resized = self.resize(values, h)
                    target_C[np.where(target_C)] = resized
                    self.C[target] = target_C
                    pass
                else:

                    pass
                t_span = [sol.t[-1], t_end]

            else:
                break  # No event triggered: finished



    def grow_ivp(self, run_id, update=True, plot=False, savefigs=True, repeat=False, set_y0=np.array(False),
                 t_end=1000, resolution=0.1, report=True):
        y0 = self.prepare_1D_array(repeat=repeat, set_y0=set_y0)
        t_start = 0
        num_points = int(t_end / resolution)
        t_span = [t_start, t_end]
        all_t = np.array([])
        all_y = np.array([]).reshape(len(y0), 0)

        self.last_event_time = 0
        self.first_invasion_copy = self.first_invasion

        if self.invade:
            t_span = [t_start, self.first_invasion_copy]
            self.next_invasion_time = self.first_invasion_copy
            self.number_of_invasions = 0
            while t_span[0] < t_end:
                while t_span[0] < self.next_invasion_time:
                    self.MCI = []
                    self.syntroph_potential = []
                    self.realised_syntrophy = []
                    sol = solve_ivp(self.time_development, t_span, y0, events=self.event,
                                    method="LSODA", atol=1e-5)  # t_eval=np.linspace(t_start, t_end, num_points), if we want to specify resolution , args=("ivp",)
                    if all_t.size == 0:  # If result arrays are empty still, populate them with the first results
                        all_t = sol.t[1::10000]
                        all_y = sol.y[:, 1::10000]
                        all_MCI = self.MCI[1::10000]
                        all_sp = self.syntroph_potential[1::10000]
                        all_rs = self.realised_syntrophy[1::10000]
                    else:  # If they are not empty, concatenate the previous and the current results
                        all_t = np.concatenate((all_t, sol.t[1::10000]), axis=0)
                        all_y = np.concatenate((all_y, sol.y[:, 1::10000]), axis=1)
                        all_MCI = np.concatenate((all_MCI, self.MCI[1::10000]), axis=0)
                        all_sp = np.concatenate((all_sp, self.syntroph_potential[1::10000]), axis=0)
                        all_rs = np.concatenate((all_rs, self.realised_syntrophy[1::10000]), axis=0)

                    # Check if an event was triggered and update the initial condition and t_span accordingly
                    if sol.t_events[0].size > 0:
                        self.last_event_time = sol.t[-1]
                        # print("event triggered")
                        y0 = np.concatenate((np.where(sol.y[:self.n_players, -1] < self.cutoffvalue, 0, sol.y[:self.n_players, -1]),
                                             sol.y[self.n_players:, -1]))
                        self.present_species = np.where(y0[:self.n_players])
                        print("t: ", sol.t[-1])
                        t_span = [sol.t[-1], t_end]
                    else:
                        break  # No event triggered: finished


                y0 = np.concatenate((np.where(sol.y[:self.n_players, -1] < self.cutoffvalue, 0, sol.y[:self.n_players, -1]),
                                     sol.y[self.n_players:, -1]))
                y0[self.n_players - self.n_invaders + self.number_of_invasions] = rng.uniform(low=1, high=10)
                self.present_species = np.where(y0[:self.n_players])
                self.next_invasion_time += self.t_invaders
                t_span = [sol.t[-1], self.next_invasion_time]
                if self.number_of_invasions > 0 and self.number_of_invasions <= self.n_invaders:
                    # what
                    if sol.y[self.n_species + self.number_of_invasions - 1, -1] > self.cutoffvalue: # only first subpop is examined
                        self.invaded[self.n_species + self.number_of_invasions - 1] = "Yes"
                    else:
                        self.invaded[self.n_species + self.number_of_invasions - 1] = "No"
                if self.number_of_invasions <= self.n_invaders:
                    self.number_of_invasions += 1
                else:
                    self.next_invasion_time = t_end

        else:  ### Many things might not work for this case such as MCI, reduced memory, check it!
            while t_span[0] < t_end:
                sol = solve_ivp(self.time_development, t_span, y0, events=self.event,
                                method="LSODA", atol=1e-5)  # t_eval=np.linspace(t_start, t_end, num_points), if we want to specify resolution , args=("ivp",)
                if all_t.size == 0:  # If result arrays are empty still, populate them with the first results
                    all_t = sol.t
                    all_y = sol.y
                else:  # If they are not empty, concatenate the previous and the current results
                    all_t = np.concatenate((all_t, sol.t), axis=0)
                    all_y = np.concatenate((all_y, sol.y), axis=1)

                # Check if an event was triggered and update the initial condition and t_span accordingly
                if sol.t_events[0].size > 0:
                    self.last_event_time = sol.t[-1]
                    y0 = np.concatenate((np.where(sol.y[:self.n_players, -1] < self.cutoffvalue, 0, sol.y[:self.n_players, -1]),
                                         sol.y[self.n_players:, -1]))
                    self.present_species = np.where(y0[:self.n_players])
                    print("t: ", sol.t[-1])
                    t_span = [sol.t[-1], t_end]
                else:
                    break  # No event triggered: finished

        self.survived = []
        for species in range(self.n_players):
            if all_y[species, -1] > self.cutoffvalue:
                self.survived.append(True)
            else:
                self.survived.append(False)

        print("survived", self.survived)


        if update:
            end_result = all_y[:, -1]
            self.communities[0].species = end_result[:self.n_players]
            self.communities[0].resources = end_result[self.n_players:]
        if report:
            self.create_report(sol.y, run_id)
        if plot:
            plotobject = Object()
            plotobject.t = all_t
            print(f"length: {len(all_t)}")
            plotobject.y = all_y
            plotobject.MCI = all_MCI
            plotobject.rs = all_rs
            plotobject.sp = all_sp
            self.plot_generation(plotobject, savefigs, run_id)
            self.plot_graph(np.where(all_y[:self.n_players, -1] > self.cutoffvalue)[0], run_id)  # checks only the 1st

    def create_report(self, sol, run_id):
        species_names = []
        for i in range(self.n_players):
            species_names.append(f'species{i}')
        species_data = {}
        for name, family, maintenance, n_reactions, n_splits, matrix, end_abundance, invaded in zip(species_names, self.family_ids[self.players],
                                                                            [round(x, 3) for x in self.all_m[self.players]], self.n_reactions,
                                                                            self.n_splits, self.C, [round(y, 3) for y in sol[:, -1]], self.invaded):
            species_data[name] = {
                'family': family,
                'maintenance': maintenance,
                'n_reactions': n_reactions,
                'n_splits': n_splits,
                'matrix': matrix,
                'end_abundance': end_abundance,
                'invaded': invaded
            }

        if self.invade:
            invader_string = "<p>There are {len(self.invaders)} invaders, first invading at t={self.first_invasion_time}, then each time after {self.t_invaders} units of time has passed</p>"
        else:
            invader_string = "<p>There are no invaders</p>"

        line1 = f"<p>Total species: {self.n_players}, Number of families: {self.n_families}, Number of resources: {self.n_resources}</p>"
        line2 = f"<p>Rate of unidirectional flow / bowel movement: {self.inflow_rate}</p>"
        line3 = f"<p>Cutoff value: {self.cutoffvalue}, h: {self.h}</p>"

        html_string = f"""
        <html>
        <head>
        <title>Species Report</title>
        <style>
            table, th, td {{
                border: 1px solid black;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 5px;
                text-align: left;
            }}
        </style>
        </head>
        <body>
        <h2>Species Report</h2>
        {line1}
        {line2}
        {line3}
        {invader_string}
        {self.d_matrix_html}
        <table>
        <tr>
            <th>Species</th>
            <th>Family</th>
            <th>Maintenance</th>
            <th>Number of Reactions</th>
            <th>Number of Splits</th>
            <th>Invaded?</th>
            <th>Survived?</th>
            <th>Matrix</th>
            <th>Optimal pH</th>
        </tr>
        """

        # Add rows to the HTML for each species
        for name, attributes in species_data.items():
            # Generate HTML for the matrix with colored cells for non-zero values
            matrix_html = ('<tr>' + ''.join(matrix_cell_html(cell) for cell in row) + '</tr>' for row in attributes['matrix'])
            matrix_html_string = '<table>' + ''.join(matrix_html) + '</table>'
            invaded_html = str(attributes.get('invaded', 'N/A'))

            html_string += f"""
            <tr>
                <td>{name}</td>
                <td>{attributes['family']}</td>
                <td>{attributes['maintenance']}</td>
                <td>{attributes['n_reactions']}</td>
                <td>{attributes['n_splits']}</td>
                <td>{attributes['invaded']}</td>
                <td>{attributes['end_abundance']}</td>
                <td>{matrix_html_string}</td>
            </tr>
            """

        # Close the HTML tags
        html_string += """
        </table>
        </body>
        </html>
        """

        # Save the HTML to a file
        html_file = f"{self.results_path}/{run_id}species_report.html"
        with open(html_file, 'w') as f:
            f.write(html_string)
        #webbrowser.open(f'http:/localhost:8000/{html_file}')

    def add_edges_from_matrix(self, G, matrix, matrix_node_name):
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] != 0:
                    G.add_edge(matrix_node_name, f"Resource_{i}", color='green')
                    G.add_edge(f"Resource_{j}", matrix_node_name, color='red')

    def plot_graph(self, species_ids, run_id):
        production = np.ones(self.n_resources)
        for i in range(self.n_players):
            production += np.multiply(self.D * self.C[i] * self.communities[0].species[i], self.communities[0].resources).sum(axis=1)
        production += self.alpha
        production *= 0.001
        species_sizes = np.repeat(500, len(species_ids))
        sizes = np.concatenate((production, species_sizes))

        G = nx.DiGraph()
        for i in range(self.n_resources):
            G.add_node(f"Resource_{i}")

        for idx in species_ids:
            G.add_node(f"Species_{idx}")
            self.add_edges_from_matrix(G, self.C[idx], f"Species_{idx}")

        edges = G.edges()
        colors = [G[u][v]['color'] for u, v in edges]

        pos = {f"Resource_{i}": (i, 1) for i in range(self.n_resources)}
        pos.update({f"Species_{i}": (idx * self.n_resources / len(species_ids), 0) for idx, i in enumerate(species_ids)})

        # Draw the graph
        plt.figure(figsize=(12, 9))
        nx.draw(G, pos, with_labels=True, node_size=sizes, node_color="lightblue", font_size=12, edge_color=colors)
        plt.savefig(f"{self.results_path}/{run_id}graph.png")
        plt.close()
        print(f"{self.results_path}/{run_id}graph.png")

    def get_invader_families(self):
        return self.family_ids[self.players[self.invaded == "Yes"]]

    def get_survivor_families(self):
        return self.family_ids[self.players[self.survived]]

    def plot_generation(self, sol, savefigs, run_id):
        cmap = plt.colormaps.get_cmap('tab20')
        color_map = {i: cmap(i) for i in range(self.n_families)}

        family_ids = self.family_ids[self.players]
        unique_family_ids = np.unique(family_ids)
        gen_spec = ["gen" if self.n_reactions_fam[i] == int(self.n_resources/5) else "spec" for i in range(self.n_families)]

        ### FIG 1.
        fig1, loc1 = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
        added_labels = set()
        for i in range(self.n_players):
            label = family_ids[i]
            if label not in added_labels:
                loc1[0].plot(sol.t, sol.y[i, :], color=color_map[label])
                added_labels.add(label)
            else:
                loc1[0].plot(sol.t, sol.y[i, :], color=color_map[label])
        loc1[0].set_title("Species abundance over time")
        loc1[0].set_yscale("log")
        legend_handles = [matplotlib.patches.Patch(color=color_map[id], label=f'{id}_{gen_spec[int(id)]}') for id in unique_family_ids]
        loc1[0].legend(handles=legend_handles)
        for i in range(self.n_players, self.n_players + self.n_resources):
            loc1[1].plot(sol.t, sol.y[i, :])
        loc1[1].set_title("Resource abundance over time")
        loc1[1].set_yscale("log")
        fig1.suptitle(f'No. of existing resources: {self.n_resources}\n'
                      f'No. of species in a subpop.: {self.n_species}, No. of different families: {self.n_families}\n'
                      f'Inflow rate: {self.inflow_rate}, Cutoff: {self.cutoffvalue}')
        if savefigs:
            fig1.savefig(
                f'{self.results_path}/{run_id}_fig1')
        plt.close()

        ### FIG 2.

        ### FIG 3.
        fig3, loc3 = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
        sol_s = sol.y[:self.n_players, :]
        for type in ["gen", "spec"]:
            mask = [True if gen_spec[int(self.family_ids[x])] == type else False for x in self.players]
            species_data = sol_s[mask, :]
            total_abundance = species_data.sum(axis=0)
            loc3[0].plot(sol.t, total_abundance, label=f"Type: {type}")
        # loc3[0].legend()
        loc3[0].set_title("Total abundance of species by type")
        loc3[0].set_yscale("log")
        for family_id in unique_family_ids:
            mask = self.family_ids[self.players] == family_id
            families_data = sol_s[mask, :]
            total_abundance = families_data.sum(axis=0)
            loc3[1].plot(sol.t, total_abundance, label=f"Family ID: {family_id}")
        loc3[1].legend()
        loc3[1].set_title("Total abundance of families")
        loc3[1].set_yscale("log")
        fig3.suptitle(f'No. of existing resources: {self.n_resources}\n'
                      f'No. of species in a subpop.: {self.n_species}, No. of different families: {self.n_families}\n'
                      f'Inflow rate: {self.inflow_rate}, Cutoff: {self.cutoffvalue}')
        if savefigs:
            fig3.savefig(
                f'{self.results_path}/{run_id}_fig3')
        plt.close()

        print(len(sol.t), "t")
        print(len(sol.MCI), "MCI")
        print(len(sol.rs), "RS")
        print(len(sol.sp), "SP")
        shortest = min(len(sol.t), len(sol.MCI), len(sol.rs), len(sol.sp))
        sol.MCI = sol.MCI[:shortest]
        sol.rs = sol.rs[:shortest]
        sol.sp = sol.sp[:shortest]
        sol.t = sol.t[:shortest]
        ### FIG 4.
        fig4, loc4 = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))
        loc4.plot(sol.t, sol.MCI)
        loc4.set_title("MCI")
        if savefigs:
            fig4.savefig(
                f'{self.results_path}/{run_id}_fig4')
        plt.close()
        ### FIG 5.
        fig5, loc5 = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))
        loc5.plot(sol.t, sol.rs)
        loc5.set_title("Relative Syntrophy")
        if savefigs:
            fig5.savefig(
                f'{self.results_path}/{run_id}_fig5')
        plt.close()
        ### FIG 6.
        fig6, loc6 = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))
        loc6.plot(sol.t, sol.sp)
        loc6.set_title("Syntrophic Potential")
        if savefigs:
            fig6.savefig(
                f'{self.results_path}/{run_id}_fig6')
        plt.close()
        np.save(f'{self.results_path}/{run_id}_results.npy', sol_s)
        #np.savetxt(f'{self.results_path}/{run_id}_results.csv', sol_s, delimiter=',', fmt='%s')

        np.save(f'{self.results_path}/{run_id}_end_results.npy', sol_s[:, -1])
        np.savetxt(f'{self.results_path}/{run_id}_end_results.csv', sol_s[:, -1], delimiter=',', fmt='%s')


foo = GCRM(n_species=1, n_resources=10, spec=0.5, spec_var=0.05, n_families=10, inflow_rate=0.1,
           results_path=folder_name, dcase="nullmodel", mcase=None, cutoffvalue=0.0001,
           m_val=0.1, invade=True, t_invaders=25, first_invasion=25, n_invaders=50, h=2)

fams = []
survived = []

current_time = datetime.now()
run_id = current_time.strftime("%Y%m%d_%H%M%S")
foo.grow_ivp(run_id=run_id, plot=True, savefigs=True, t_end=1000, resolution=0.5)
fams.append(foo.get_invader_families())
survived.append(foo.get_survivor_families())

for i in range(10):
    print(f"Run {i}")
    current_time = datetime.now()
    run_id = current_time.strftime("%Y%m%d_%H%M%S")
    foo.grow_ivp(run_id=run_id, plot=True, savefigs=True, t_end=1000, resolution=0.5, repeat=True)
    fams.append(foo.get_invader_families())
    survived.append(foo.get_survivor_families())


with open(f"{folder_name}/invaded_families.csv", mode='w', encoding='utf-8') as file:
    for row in fams:
        # Create a string for the row
        row_string = ','.join(str(int(item)) for item in row)
        # Write the string to the file, followed by a newline
        file.write(row_string + '\n')

with open(f"{folder_name}/survived_families.csv", mode='w', encoding='utf-8') as file:
    for row in survived:
        # Create a string for the row
        row_string = ','.join(str(int(item)) for item in row)
        # Write the string to the file, followed by a newline
        file.write(row_string + '\n')



