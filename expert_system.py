import collections
import collections.abc

collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence

from experta import Fact
from rules import HeartDiseaseExpert



# Engine

engine = HeartDiseaseExpert()
engine.reset()

print("\n Heart Disease Expert System\n")


# User Input (ALL FEATURES)

data = {
    "cp": int(input("Chest Pain Type (cp): ")),
    "thalach": int(input("Max Heart Rate (thalach): ")),
    "slope": int(input("Slope: ")),
    "age": int(input("Age: ")),
    "sex": int(input("Sex (0/1): ")),
    "thal": int(input("Thal: ")),
    "ca": int(input("Number of vessels (ca): ")),
    "oldpeak": float(input("Oldpeak: ")),
    "exang": int(input("Exercise Angina (0/1): "))
}


# Feed Facts

for k, v in data.items():
    engine.declare(Fact(**{k: v}))

# Start rule
engine.declare(Fact(start="run"))

# Run Engine

engine.run()