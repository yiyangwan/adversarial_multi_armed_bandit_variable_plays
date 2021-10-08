import matplotlib.pyplot as plt
import numpy as np

# matplotlib.rcParams['text.usetex'] = True
plt.rcParams.update({"font.size": 12})

x = np.linspace(0.0, 1.0, num=20)
N = 10
a = 2
b = 4
y1 = [xx * (N - a) / N for xx in x]
y2 = [xx * (N - b) / N for xx in x]


fig = plt.figure()


ax = plt.gca()
ax.grid(True)
ax.plot(x, y1, color="blue")
ax.plot(x, y2, color="red")
ax.fill_between(x, y1, y2)
ax.plot(x, [(N - a) / N for _ in x], linestyle="--", color="blue")
ax.plot(x, [(N - b) / N for _ in x], linestyle="--", color="red")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_xlabel("Success rate of attack")
ax.set_ylabel(r"$\bar{r}_{\infty}$ Average reward of the attacker")

ax.set_yticks([0.6, 0.8, 1], [r"$\frac{N-b}{N}$", r"$\frac{N-a}{N}$", "1"])

fig.savefig("reward_vs_attack_rate.png")

_ = fig.show()
