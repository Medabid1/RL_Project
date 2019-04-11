import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 
sns.set()

no_her = []
with open('/home/mohamed/no her', 'r') as f :
    for line in f :
        no_her.append(float(line.split('is ')[1][:4] )*10)

x = np.linspace(0, 200 , 200)
her = [0.00] + [10.00 for i in range(199)]

#print(her , no_her)
plt.plot(x, her)
plt.plot(x, no_her, 'r')
plt.legend(['DDPG with her', 'DDPG'])
plt.show()