class Matcher():
    def __init__(self):
        pass

    def find_matches(self, query, pattern, n_matches=1):
        """
        :param query: (ndarray) image to find matches on
        :param pattern: (ndarray) pattern to match
        :return: (list) location of matches
        """
        raise NotImplementedError