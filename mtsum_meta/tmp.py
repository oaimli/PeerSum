import random
r1 = "police killed the gunman."
r2 = "the gunman was shot down by police."
c1 = "police ended the gunman."
c2 = "the gunman murdered police."
# random.seed(42)
x = [r1, r2, c1, c2]
random.shuffle(x)
print(x)