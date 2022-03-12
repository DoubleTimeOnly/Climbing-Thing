import cv2
from climbing_thing.utils import logger
from climbing_thing.patternmatching import sift


log = logger.get_logger(__name__)
log.setLevel(logger.DEBUG_WITH_IMAGES)

STRATEGIES = {
    "sift": sift.SIFTMatcher,
}

class PatternMatcher:
    def __init__(self, path_to_pattern, strategy: str= "sift"):
        '''
        :param screen_dimensions: (tuple / list) cornerX, cornerY, width, height
        :param strategy: (string) the pattern matching algorithm to use
        '''
        self.load_pattern(path_to_pattern)
        self.strategy = STRATEGIES[strategy.lower()]()

    def find_pattern(self, query, n_matches=1):
        if self.pattern is None:
            raise ValueError("Pattern is None. Likely because it has not been loaded yet.")
        matched_patterns = self.strategy.find_matches(query, self.pattern, n_matches=n_matches)
        return matched_patterns

    def load_pattern(self, path_to_pattern):
        log.debug(f"Reading pattern: {path_to_pattern}")
        self.pattern = cv2.imread(path_to_pattern, 1)
        if self.pattern is None:
            raise FileNotFoundError(f"Could not load file {path_to_pattern}")
        if log.level <= logger.DEBUG_WITH_IMAGES:
            cv2.imshow("loaded pattern", self.pattern)
            cv2.waitKey(0)
