"""Explicit 3:1 human/corrective sample scheduling; no training launcher."""

import hashlib
import json
import random
from collections import OrderedDict, defaultdict

try:
    from .bridge import encode_sample
except ImportError:
    from bridge import encode_sample


SCHEDULE = ("human", "human", "human", "corrective")


class ThreeToOneProvider:
    """Repeat H,H,H,C: an accumulation of eight has exactly six H and two C.

    Both children implement next_batch/state_dict/load_state_dict. Diagnostic
    identities are returned separately from model inputs; they are never fed
    into the policy. Checkpoints remain the caller's step-boundary concern.
    """

    def __init__(self, human, corrective):
        self.providers = {"human": human, "corrective": corrective}
        self.draws = 0

    def next_batch(self):
        source = SCHEDULE[self.draws % len(SCHEDULE)]
        inputs, actions, valid, identity = self.providers[source].next_batch()
        self.draws += 1
        return inputs, actions, valid, {"source": source, "mixture_draw": self.draws, "sample": identity}

    def state_dict(self):
        return {"format_version": 1, "schedule": list(SCHEDULE), "draws": self.draws,
                "providers": {name: provider.state_dict() for name, provider in self.providers.items()}}

    def load_state_dict(self, state):
        if (state.get("format_version") != 1 or state.get("schedule") != list(SCHEDULE) or
                type(state.get("draws")) is not int or state["draws"] < 0 or
                set(state.get("providers", {})) != set(self.providers)):
            raise ValueError("Mixture checkpoint has an incompatible schedule/state")
        for name, provider in self.providers.items():
            provider.load_state_dict(state["providers"][name])
        self.draws = state["draws"]


class CorrectivePool:
    """Task-uniform, recording-uniform, eligible-action-step-uniform sampling.

    Readers must already have passed the successful-continuation gate. Their
    inventory includes the exact parent/continuation manifests and sample
    range. Failed prefixes and pre-intervention context are never supervision.
    """

    def __init__(self, readers, processor, device, seed):
        from .corrective_reader import CorrectiveBranchReader, LiveInterventionReader
        if not readers:
            raise ValueError("At least one verified corrective reader is required")
        self.readers, self.processor, self.device = list(readers), processor, device
        self.by_task = defaultdict(list)
        inventories = []
        for index, reader in enumerate(self.readers):
            if not isinstance(reader, (CorrectiveBranchReader, LiveInterventionReader)):
                raise TypeError("Corrective pool requires the source-verified branch reader")
            inventory = reader.inventory()
            if (inventory.get("source") not in {"verified_successful_continuation", "verified_live_intervention_success"} or
                    type(inventory.get("steps")) is not int or inventory["steps"] < 1):
                raise ValueError("Corrective pool only accepts verified nonempty successful continuations")
            inventories.append(inventory)
            self.by_task[inventory["task"]].append(index)
        identities = [json.dumps(item, sort_keys=True) for item in inventories]
        if len(set(identities)) != len(identities):
            raise ValueError("Duplicate corrective branch entries")
        self.identity = hashlib.sha256(json.dumps(inventories, sort_keys=True).encode()).hexdigest()
        self.tasks = sorted(self.by_task)
        self.random = random.Random(seed)
        self.draws = 0
        self.active_readers = OrderedDict()

    def next_batch(self):
        task = self.tasks[self.random.randrange(len(self.tasks))]
        choices = self.by_task[task]
        index = choices[self.random.randrange(len(choices))]
        reader = self.readers[index]
        self.active_readers[index] = True
        self.active_readers.move_to_end(index)
        while len(self.active_readers) > 2:
            previous, _ = self.active_readers.popitem(last=False)
            self.readers[previous].clear_caches()
        timestep = self.random.randrange(len(reader))
        sample = reader.sample(timestep)
        inputs, actions, valid = encode_sample(sample, self.processor, self.device)
        self.draws += 1
        return inputs, actions, valid, {"task": task, "recording": str(reader.root),
                                       **reader.sample_identity(timestep), "draw": self.draws}

    def state_dict(self):
        return {"format_version": 1, "identity": self.identity, "rng": self.random.getstate(), "draws": self.draws}

    def load_state_dict(self, state):
        if state.get("format_version") != 1 or state.get("identity") != self.identity:
            raise ValueError("Corrective pool checkpoint belongs to a different branch selection")
        if type(state.get("draws")) is not int or state["draws"] < 0:
            raise ValueError("Invalid corrective sample count")
        self.random.setstate(state["rng"])
        self.draws = state["draws"]
