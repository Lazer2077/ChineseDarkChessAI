"""Opt-in root determinization bridge; original root observations stay intact.

The value/policy encoder is unchanged and does not observe the posterior. This
is a search sampling experiment, not a belief-aware critic or a ReBeL solver.
"""
from dataclasses import fields
from adapters.yuan_observer import load_sources

jq, im = load_sources()


class InventoryInformationState(im.InformationState):
    __slots__ = ('inventory', 'observer')

    def determinize(self, rng):
        if self.inventory.uniform:
            # Exact control: no additional random draws, same upstream sampling.
            return super().determinize(rng)
        h = self.inventory.sample(rng)
        data = {field.name: getattr(self, field.name) for field in fields(im.InformationState)}
        data['pool_known'] = [row.copy() for row in self.pool_known]
        data['pool_total_known'] = self.pool_total_known.copy()
        data['unknown_removals'] = self.unknown_removals.copy()
        for t, n in zip(jq.POOL_TYPES, h):
            data['pool_known'][self.observer][t] -= n
        data['pool_total_known'][self.observer] = self.inventory.live
        data['unknown_removals'][self.observer] = 0
        return im.InformationState(**data).determinize(rng)


class InventoryRootBelief:
    def __init__(self, belief, root_state, observer, inventory):
        if observer != root_state.side:
            raise ValueError('This bridge serves only the root actor')
        source = belief.info_state(root_state, observer)
        if tuple(source.pool_known[observer][t] for t in jq.POOL_TYPES) != inventory.public:
            raise ValueError('Public inventory mismatch')
        if source.unknown_removals[observer] != inventory.captured or source.pool_total_known[observer] != inventory.live:
            raise ValueError('Capture count mismatch')
        self.info = InventoryInformationState(**{f.name: getattr(source, f.name) for f in fields(im.InformationState)})
        self.info.inventory, self.info.observer = inventory, observer
        self.root_state, self.observer = root_state, observer

    def info_state(self, state, observer):
        if state is not self.root_state or observer != self.observer:
            raise ValueError('Never query another actor private ledger')
        return self.info
