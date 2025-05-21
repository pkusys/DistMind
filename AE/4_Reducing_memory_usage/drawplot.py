from matplotlib import pyplot as plt
import numpy as np

output_dir = "./4_Reducing_memory_usage/fig10.png"

x = [1.0, 2.0, 5.0] + np.linspace(10, 100, 10).tolist()
x = np.array(x)
distMind = x * 4.0
pipeswitch32 = x * 32.0
pipeswitch1T = x * 1000.0

fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.set_size_inches(6, 3)

plt.plot(x, distMind, label='DistMind', marker='D')
plt.plot(x, pipeswitch32, label='PipeSwitch(32GB)', marker='|')
plt.plot(x, pipeswitch1T, label='PipeSwitch(1TB)', marker='v')

plt.xlabel('Number of servers')
plt.ylabel('Host Memory (GB)')
plt.yscale('log', subs=[])
plt.ylim(1, 1000000)
plt.xlim(0, 100)
plt.margins(0.2)
plt.legend(loc='upper left', ncol=2, shadow=False)
plt.title('Fig. 10. Required host memory for different systems.')
plt.savefig(output_dir, bbox_inches='tight')
