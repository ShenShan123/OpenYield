class Memory:
    def __init__(self):
        self.actions = []
        self.states_spec = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states_spec[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]