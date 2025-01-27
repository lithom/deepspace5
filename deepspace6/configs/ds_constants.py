class DeepSpaceConstants:
    def __init__(self, MAX_ATOMS=32, MAX_BONDS=64, MAX_NEIGHBORS=4, ADJACENCY_POS_WEIGHT= (4*64) / 2.0 , MAX_GRAPH_DIST_EXACT=6, DIST_SCALING=32.0, MAX_SMALL_RING=6, MAX_LARGE_RING_TEST=24,
                 GEOM_MAX_MEAN= 25.0,GEOM_MIN_VAR=0.0001,GEOM_MAX_VAR=15.0,device="cuda"):
        self.MAX_ATOMS = MAX_ATOMS
        self.MAX_BONDS = MAX_BONDS
        self.MAX_NEIGHBORS = MAX_NEIGHBORS
        self.ADJACENCY_POS_WEIGHT = ADJACENCY_POS_WEIGHT
        self.MAX_GRAPH_DIST_EXACT = MAX_GRAPH_DIST_EXACT
        self.DIST_SCALING = DIST_SCALING
        self.MAX_SMALL_RING = MAX_SMALL_RING
        self.MAX_LARGE_RING_TEST = MAX_LARGE_RING_TEST
        self.atom_types = ["C","N","O","F","P","S","Cl","Br","I"]

        self.GEOM_MAX_MEAN = GEOM_MAX_MEAN
        self.GEOM_MAX_VAR  = GEOM_MAX_VAR
        self.GEOM_MIN_VAR  = GEOM_MIN_VAR

        self.device = device  # 'cuda' or 'cpu'

    @classmethod
    def from_config(cls, config):
        """
        Create an instance from a configuration dictionary.
        :param config: Dictionary containing configuration parameters.
        :return: DeepSpaceConstants instance
        """
        return cls(
            MAX_ATOMS=config.get("MAX_ATOMS", 32),
            MAX_BONDS=config.get("MAX_BONDS", 64),
            MAX_NEIGHBORS=config.get("MAX_NEIGHBORS", 4),
            ADJACENCY_POS_WEIGHT=config.get("ADJACENCY_POS_WEIGHT",128),
            DIST_SCALING=config.get("DIST_SCALING", 32.0),
            MAX_SMALL_RING=config.get("MAX_SMALL_RING",6),
            MAX_LARGE_RING_TEST=config.get("MAX_LARGE_RING_TEST", 24),
            GEOM_MIN_VAR=config.get("GEOM_MIN_VAR",0.0001),
            GEOM_MAX_VAR=config.get("GEOM_MAX_VAR", 15.0),
            GEOM_MAX_MEAN=config.get("GEOM_MAX_MEAN", 25.0),
            device=config.get("device", "cuda"),
        )
