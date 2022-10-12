# neural-networks-hands-on

```python

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
```

Let's define two normal variables x0 and x1:

* x0 is a normal centered in -1,5
* x1 is a normal centered in 1,5

Both with standard deviation 1,
100 samples of each

Our challenge will be to decide on new inputs if the correspond to the Random Variable X0 or X1

```python
normal = np.random.normal
sigma = 1
x0 = normal(-1.5, sigma, 100)
x1 = normal(1.5, sigma, 100)
y0 = np.zeros_like(x0)
y1 = np.ones_like(x1)
# LEt's visualize it
plt.xlim(-5,5)
plt.plot(x0, y0,'o')
plt.plot(x1, y1,'o')
plt.xlabel('x')
plt.ylabel('category')
# Probably we will see may X0 samples closer to X1 center and viceversa
plt.clf()
plt.xlim(-5,5)
plt.hist(x0,bins=50, range=(-5,5), alpha=0.5)
plt.hist(x1,bins=50, range=(-5,5), alpha=0.5)
plt.xlabel('x')
plt.ylabel('counts')
# visually in x = 0 between -1.5 and 1.5 would be a good separation boundary
```