import numpy as np

data = np.genfromtxt(
    "data/fantasy_names.txt",
    delimiter=" ",
    dtype=[("name", "S50"), ("race", "S1")],
    converters={0: lambda s: s.upper()},
)

names = data["name"]
races = data["race"]
