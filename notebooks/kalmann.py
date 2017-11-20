import numpy as np
import matplotlib.pyplot as plt
import random

A = B = H = 1

MU = 0

# apriori sigma
SIGMA = 0
#abweichung dreckstger shit
SIGMA_DIRTIER = 2.0 ** 2
#abweichung dreckig
SIGMA_DIRTY = 0.2 ** 2
VALUES = 100

clean = []
clean.extend(range(1,VALUES+1))

dirty = []
dirtier = []
difference = []
corrected = []


def obscure_data(data, amount):
	new_data = []
	random_amount = 0;
	for d in data:
		random_amount = random.randint(-1, 1) * amount
		new_data.append(d + random_amount)
		#print("Changed value to {new_val} with random_amount {rd}".format(new_val=d, rd=random_amount))
	return new_data

def diff (clean, corr):
	global difference

	diff = (clean - corr)
	return diff

def plot():
	global clean, corrected, VALUES, difference, corrected

	x = np.linspace(0, VALUES, VALUES)
	plt.figure()
	plt.title('pred vs real vs dirtier vs dirty')
	plt.scatter(x, difference, label='diff')
	plt.legend()
	plt.show()


def kalman(index, change):
	global SIGMA, SIGMA_DIRTIER, A, B, MU

	# pred
	apriori_val = A * MU + B * change
	apriori_sigma = A * SIGMA * A + SIGMA_DIRTIER

	# corr
	gain = apriori_sigma/(apriori_sigma + SIGMA_DIRTY)
	aposteriori_val = apriori_val + (gain * (dirty[index] - apriori_val))

	# for next loop
	MU = aposteriori_val
	# print(SIGMA)
	SIGMA = (1 - gain) * apriori_sigma
	# print(SIGMA)
	return aposteriori_val



dirty = obscure_data(clean, 0.2)
dirtier = obscure_data(clean, 2.0)

for key, val in enumerate(clean):
	kalmaned_change = kalman(key, 1)
	new_val = clean[key] + kalmaned_change
	corrected.append(kalmaned_change)
	difference.append(diff(val, kalmaned_change))

    #print("original {} vs kalmaned {} | diff {}".format(val, kalmaned_change, diff(val, kalmaned_change)))


plot()

def kalmann2(sequence):
    # intial parameters
    n_iter = len(sequence)
    sz = (n_iter,) # size of array
    #x = -0.37727 # truth value (typo in example at top of p. 13 calls this z)
    z = sequence #np.random.normal(x,0.1,size=sz) # observations (normal about x, sigma=0.1)

    Q = 1e-5 # process variance

    # allocate space for arrays
    xhat=np.zeros(sz)      # a posteri estimate of x
    P=np.zeros(sz)         # a posteri error estimate
    xhatminus=np.zeros(sz) # a priori estimate of x
    Pminus=np.zeros(sz)    # a priori error estimate
    K=np.zeros(sz)         # gain or blending factor

    R = 0.1**4 # estimate of measurement variance, change to see effect

    # intial guesses
    xhat[0] = sequence[0]
    P[0] = 1.0

    for k in range(1,n_iter):
        # time update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1]+Q

        # measurement update
        K[k] = Pminus[k]/( Pminus[k]+R )
        xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
        P[k] = (1-K[k])*Pminus[k]
    return xhat

plt.plot(dirtier)
plt.show()
new = kalmann2(dirtier)
plt.plot(new)
plt.show()
