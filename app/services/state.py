class GlobalState:
    def __init__(self):
        self.model = None
        self.df = None
        self.module_embeddings = None

    def is_ready(self):
        return self.model is not None and self.df is not None and self.module_embeddings is not None

# Singleton instance
state = GlobalState()
