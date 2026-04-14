import collections
import collections.abc

# Fix for Python 3.10+
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Sequence = collections.abc.Sequence

from experta import *

class HeartDiseaseExpert(KnowledgeEngine):

    def __init__(self):
        super().__init__()
        self.risk = 0

    # HIGH RISK RULES
    @Rule(Fact(age=P(lambda x: x > 55)))
    def high_age(self):
        self.risk = 1

    @Rule(Fact(thalach=P(lambda x: x < 100)))
    def low_hr(self):
        self.risk = 1

    @Rule(Fact(oldpeak=P(lambda x: x > 2)))
    def high_oldpeak(self):
        self.risk = 1

    @Rule(Fact(exang=1))
    def angina(self):
        self.risk = 1

    @Rule(Fact(ca=P(lambda x: x > 1)))
    def vessels(self):
        self.risk = 1

    @Rule(Fact(cp=0))
    def chest_pain(self):
        self.risk = 1

    @Rule(Fact(slope=2))
    def bad_slope(self):
        self.risk = 1

    @Rule(Fact(thal=3))
    def thal_risk(self):
        self.risk = 1

    #  LOW RISK RULES
    @Rule(Fact(age=P(lambda x: x < 40)))
    def young(self):
        self.risk = 0

    @Rule(Fact(thalach=P(lambda x: x > 150)))
    def good_hr(self):
        self.risk = 0

    @Rule(Fact(oldpeak=P(lambda x: x < 1)))
    def normal_ecg(self):
        self.risk = 0

    @Rule(Fact(cp=2))
    def normal_cp(self):
        self.risk = 0

    #  FINAL RESULT
    @Rule(Fact(start="run"))
    def result(self):
        if self.risk == 1:
            print("\n⚠️ HIGH RISK of Heart Disease\n")
        else:
            print("\n LOW RISK of Heart Disease\n")