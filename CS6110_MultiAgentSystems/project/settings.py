from enum import Enum

class InfluencerMethod(Enum):
    RANDOM = "Rando Placement"
    GRID = "Grid Placement"
    BORDER = "Border Placement"
    GRAPH = "Graph Placement"

 
fps: int = 65
width: int = 1500
height: int = 1000
num_boids: int = 30
num_influencers: int = 10
influencer_method = InfluencerMethod.GRAPH
render: bool = False
iterations = 60*60
tests = 5
desnity_metric = 1