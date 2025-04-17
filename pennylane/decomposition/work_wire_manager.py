from pennylane.measurements import measure


class WorkWireManager:

    def __init__(self, zeroed, burnable, borrowable, garbage):
        self._zeroed = zeroed
        self._burnable = burnable
        self._borrowable = borrowable
        self._garbage = garbage
        self._zeroed_with_uncompute = []

    @property
    def num_possible_zeroed(self):
        return len(self._zeroed) + len(self._burnable) + self._num_zeroed_with_uncompute

    @property
    def _num_zeroed_with_uncompute(self):
        return sum(len(wires) for wires, _ in self._zeroed_with_uncompute)

    def get_zeroed(self, num_wires: int):
        wires = self._zeroed[:num_wires]
        num_retrieved = len(wires)
        if num_retrieved == num_wires:
            return wires
        wires += self._burnable[: num_wires - num_retrieved]
        if len(wires) == num_wires:
            return wires

        if self._zeroed_with_uncompute:
            new_w, uncompute = self._zeroed_with_uncompute.pop()
            uncompute(new_w)
            wires += new_w
            self._zeroed += new_w
            if len(wires) == num_wires:
                return wires

        raise ValueError("Not enough work wires in the zero state.")

    def get_zeroed_with_uncompute(self, num_wires: int, uncompute):
        wires = self._zeroed[:num_wires]
        if len(wires) == num_wires:
            self._zeroed = self._zeroed[num_wires:]
            self._zeroed_with_uncompute.append((wires, uncompute))
            return wires

        raise NotImplementedError

    @property
    def num_possible_burnable(self):
        return len(self._burnable) + len(self._garbage)

    def get_burnable(self, num_wires: int):
        from_burnable = self._burnable[:num_wires]
        if len(from_burnable) == num_wires:
            self._burnable = self._burnable[num_wires:]
            self._garbage += from_burnable
            return from_burnable

        from_garbage = []
        for i in range(num_wires - len(from_burnable)):
            new_wire = self._garbage[i]
            measure(new_wire, reset=True)
            from_garbage.append(new_wire)

        if len(from_garbage) + len(from_burnable) == num_wires:
            self._burnable = self._burnable[num_wires:]
            self._garbage += from_burnable
            return from_burnable + from_garbage

        raise ValueError("Not enough work wires in burnable or garbage states.")

    @property
    def num_possible_borrowable(self):
        return len(self._garbage) + len(self._borrowable) + len(self._burnable) + len(self._zeroed)

    def get_borrowable(self, num_wires):
        wires = self._garbage[:num_wires]
        if len(wires) == num_wires:
            return wires
        wires += self._burnable[: num_wires - len(wires)]
        if len(wires) == num_wires:
            return wires
        wires += self._borrowable[: num_wires - len(wires)]
        if len(wires) == num_wires:
            return wires
        wires += self._zeroed[: num_wires - len(wires)]
        if len(wires) == num_wires:
            return wires
        raise ValueError("not enough wires available.")

    @property
    def num_possible_garbage(self):
        return len(self._garbage)

    def get_garbage(self, num_wires):
        wires = self._garbage[:num_wires]
        if len(wires) == num_wires:
            return wires
        wires += self.get_burnable(num_wires - len(wires))
        if len(wires) == num_wires:
            return wires
        raise ValueError("not enough wires available.")
