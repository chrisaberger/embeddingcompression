class Bucketer:
    def __init__(self, num_buckets, max_num_buckets):
        """
        Negative numbers are a way to force the maximum number of buckets. 
        """
        if num_buckets <= 0:
            num_buckets = max_num_buckets
        self.num_buckets = num_buckets
        self.max_num_buckets = max_num_buckets

    def extra_bytes_needed(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

    def bucket(self, X):
        raise NotImplementedError