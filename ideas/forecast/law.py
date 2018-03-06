import numpy as np
import matplotlib.pyplot as plt

# Use a "reduced" version with only the last 10^6 entries
data = np.genfromtxt('coinbaseUSD-1M.csv' , delimiter=',')

t = data[:,0] # Time in UNIX timestamp
y = data[:,1] # The value in USD of 1 BTC

dt = np.diff(t)
dy = np.diff(y)

# Remove values with dt == 0 to avoid division by 0

dy = dy[dt != 0]
dt = dt[dt != 0]

x = dy/dt

# We are interested in the absolute value of dy/dt, but it can be
# negative. So we use for example the square.

x = x ** 2
#x = np.abs(x)

# Now sort the array in decreasing order, in order to beautifully plot x

x = x[np.argsort(-x)]

plt.plot(x)

plt.title("Who I am?")
plt.ylabel('dy/dt')

# Log scale in both axis
plt.xscale('log')
plt.yscale('log')

#plt.show()
plt.savefig('speed-law.png')
