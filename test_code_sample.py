def fibonacci(n):
    """A simple Fibonacci function with poor naming conventions."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class my_cache:  # Poor naming - should be PascalCase
    def __init__(self):
        self.data = {}
    
    def get_item(self, key):
        return self.data.get(key)
    
    def set_item(self, key, value):
        self.data[key] = value

# Missing type hints and documentation
def calculate_something(x, y):
    result = x * y + fibonacci(5)
    return result