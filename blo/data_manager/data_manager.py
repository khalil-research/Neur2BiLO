from abc import ABC, abstractmethod

import time
import numpy as np
import pickle as pkl

from multiprocessing import Manager, Pool


class DataManager(ABC):
    
    @abstractmethod
    def initialize_problem(self):
        """ Initializes BLO problem. """
        pass


    @abstractmethod
    def _solve_lower_level_mp(self, x, instance, inst_id, mp_time, mp_count, n_samples):
        """ Obtains the cost of the suboptimal first stage solution and computes any additional features. """
        pass


    @abstractmethod
    def _sample_random_x(self, instance, X_hash=None):
        """ Sammples random upper-level decision.  X_hash can be used to remove duplicates. """
        pass


    @abstractmethod
    def _get_problem_data(self, cfg):
        """ Gets all problem data based on cfg.  """
        pass


    def _load_problem(self):
        """ Loads problem. """
        self.prob = pkl.load(open(self.problem_path, 'rb'))


    def generate_dataset(self, n_procs):
        """ Generate dataset for training ML models. """
        print("Generating dataset for: ")
        for k, v in self.cfg.__dict__.items():
            print(f"    {k}: {v}")

        # load problem
        self._load_problem()

        # seed
        np.random.seed(self.cfg.seed)

        total_time = time.time()

        # sample instances
        print("Sampling instances/decisions for dataset...")
        instances = []
        for i in range(self.cfg.n_samples_inst):
            instances.append(self.blo.sample_instance(self.cfg))

        # split instances to train/test
        tr_instances, val_instances = self._get_data_split(instances)

        # sample upper-level decisions for each training instance
        tr_procs_to_run = []
        for inst_id, instance in enumerate(tr_instances):
            # sampling for continous/binary decisions (kp, cng, drp)
            X_hash = set()
            for i in range(self.cfg.n_samples_per_inst):
                x = self._sample_random_x(instance, X_hash)
                tr_procs_to_run.append((instance, inst_id, x))

        # sample upper-level decisions for each validation instance
        val_procs_to_run = []
        for inst_id, instance in enumerate(val_instances):
            # sampling for continous/binary decisions (kp, cng, drp)
            X_hash = set()
            for i in range(self.cfg.n_samples_per_inst):
                x = self._sample_random_x(instance, X_hash)
                val_procs_to_run.append((instance, inst_id, x))

        print("  Done.")

        print("Solving lower level problems for each training instance...")

        # Solve all suboptimal LPs for training instances
        tr_time = time.time()
        tr_data = []

        print(f"  Running {'{:,}'.format(len(tr_procs_to_run))} with {n_procs} cpus... ")
        mp_count = Manager().Value('i', 0)
        mp_time = time.time()

        # optional.  Set to true if debugging.  Avoids any anymultiprocessing related issues.
        debug = False
        if debug:
            print("Running in debugging mode...")
            for instance, inst_id, x in tr_procs_to_run:
                res = self._solve_lower_level_mp(x, instance, inst_id, mp_time, mp_count, len(tr_procs_to_run))
            print("Successfully ran all data collection.  Exiting!")
            exit()

        pool = Pool(n_procs)
        
        for instance, inst_id, x in tr_procs_to_run:
            res = pool.apply_async(self._solve_lower_level_mp, args=(x, instance, inst_id, mp_time, mp_count, len(tr_procs_to_run)))
            tr_data.append(res)

        tr_data = list(map(lambda x: x.get(), tr_data))

        pool.close()
        pool.join()        

        tr_time = time.time() - tr_time

        print("  Done.")

        print("Solving lower level problems for each validation instance...")

        # Solve all suboptimal LPs for validation instances
        val_time = time.time()
        val_data = []

        print(f"  Running {'{:,}'.format(len(val_procs_to_run))} with {n_procs} cpus... ")
        mp_count = Manager().Value('i', 0)
        mp_time = time.time()

        pool = Pool(n_procs)
        
        for instance, inst_id, x in val_procs_to_run:
            res = pool.apply_async(self._solve_lower_level_mp, args=(x, instance, inst_id, mp_time, mp_count, len(val_procs_to_run)))
            val_data.append(res)

        val_data = list(map(lambda x: x.get(), val_data))

        pool.close()
        pool.join()        

        val_time = time.time() - val_time

        print("  Done.")

        total_time = time.time() - total_time

        # get train/validation split, then store
        ml_data = {
            "tr_data": tr_data,
            "val_data": val_data,
            "data": tr_data + val_data,
            "total_time": total_time,
            "tr_time" : tr_time,
            "val_time" : val_time,
        }

        print("Total Time:         ", total_time)
        print("Train dataset size: ", len(tr_data))
        print("Valid dataset size: ", len(val_data))

        pkl.dump(ml_data, open(self.ml_data_path, 'wb'), protocol=pkl.HIGHEST_PROTOCOL)
        

    def _get_data_split(self, instances):
        """ Gets train/validation splits for the data.  """
        # get permutation of all indicies
        perm = np.random.permutation(len(instances))

        # get indicies for train/validation instances
        split_idx = int(self.cfg.tr_split * (len(instances)))
        tr_idx = perm[:split_idx].tolist()
        val_idx = perm[split_idx:].tolist()

        # get train/validation instances
        tr_instances = [instances[i] for i in tr_idx]
        val_instances = [instances[i] for i in val_idx]

        return tr_instances, val_instances


    def update_mp_status(self, mp_count, mp_time, n_samples):
        """ Printing/status for multiprocessing. """
        mp_count.value += 1
        count = mp_count.value

        if count % 1000 == 0:
            if count == 1000:
                print("    Sample              | Percent   | Time      | ETA          ")
                print("    -----------------------------------------------------------")

            n_samples_str = '{:,}'.format(n_samples)

            pct = count / n_samples
            pct_str = '{:.2%}'.format(pct)
            
            count_str = '{:,}'.format(count)

            time_ = time.time() - mp_time
            time_str = '{:.2f}'.format(time_)

            eta_ = (time_/count) *  (n_samples - count)
            eta_str = '{:.2f}'.format(eta_)

            print(f"    {count_str} / {n_samples_str}   |   " \
                   f"{pct_str}   |   " \
                   f"{time_str}s   |   " \
                   f"{eta_str}s"
                   )