class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places

        new_val = init
        for i in range(iterations):
            new_val = new_val - learning_rate * (2*new_val)
        
        return round(new_val,5)
            

        # pass
