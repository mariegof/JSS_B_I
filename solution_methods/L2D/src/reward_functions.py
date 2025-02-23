from abc import ABC, abstractmethod

class RewardFunction(ABC):
    """Abstract base class for reward calculation strategies."""
    
    @abstractmethod
    def calculate(self, current_objective, previous_objective, operation_index=None):
        """Calculate the reward based on current and previous objectives."""
        pass

class StandardReward(RewardFunction):
    """Standard reward: negative difference between current and previous objectives."""
    
    def calculate(self, current_objective, previous_objective, operation_index=None):
        return -(current_objective - previous_objective)

class WeightedDifferenceReward(RewardFunction):
    """Weighted difference between completion time and lower bound for each operation."""
    
    def calculate(self, current_objective, previous_objective, operation_index=None):
        # Here current_objective is completion time and previous_objective is lower bound
        return -(current_objective - previous_objective)

class IndexedWeightedReward(RewardFunction):
    """Weighted difference scaled by operation index."""
    
    def calculate(self, current_objective, previous_objective, operation_index):
        base_reward = -(current_objective - previous_objective)
        operation_count = operation_index.size
        scale_factor = (operation_index + 1) / operation_count
        return base_reward * scale_factor