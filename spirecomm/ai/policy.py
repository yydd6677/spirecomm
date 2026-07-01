import importlib
import inspect


class CombatPolicy:

    source_name = "CombatPolicy"

    def choose_action(self, game_state, fallback_agent, coordinator=None):
        raise NotImplementedError()

    def reload(self):
        return None


class RuleBasedCombatPolicy(CombatPolicy):

    source_name = "RuleBasedCombatPolicy"

    def choose_action(self, game_state, fallback_agent, coordinator=None):
        return fallback_agent.get_play_card_action()


class LoadedCombatPolicy(CombatPolicy):

    def __init__(self, implementation):
        self.implementation = implementation
        self.source_name = getattr(implementation, "source_name", implementation.__class__.__name__)

    def choose_action(self, game_state, fallback_agent, coordinator=None):
        if hasattr(self.implementation, "choose_action"):
            return self.implementation.choose_action(game_state, fallback_agent, coordinator=coordinator)
        return self.implementation(game_state, fallback_agent)

    def reload(self):
        if hasattr(self.implementation, "reload"):
            return self.implementation.reload()
        return None


def load_policy(spec):
    module_name, attribute_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    target = getattr(module, attribute_name)
    if inspect.isclass(target):
        target = target()
    return LoadedCombatPolicy(target)
