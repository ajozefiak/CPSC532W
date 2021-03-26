from evaluator import evaluate
import torch
import numpy as np
import json
import sys
import distributions as dist
import matplotlib.pyplot as plt






def run_until_observe_or_end(res):
    cont, args, sigma = res
    res = cont(*args)
    while type(res) is tuple:
        if res[2]['type'] == 'observe':
            return res
        cont, args, sigma = res
        res = cont(*args)

    res = (res, None, {'done' : True}) #wrap it back up in a tuple, that has "done" in the sigma map
    return res

def resample_particles(particles, log_weights):

    n_particles = len(particles)
    new_particles = []

    weights = [torch.exp(w).tolist() for w in log_weights]
    d = dist.Categorical(torch.tensor(weights))

    for i in range(n_particles):
        sampled_idx = d.sample()
        new_particles.append(particles[sampled_idx])

    logZ = np.log(np.mean(weights))
    return logZ, new_particles



def SMC(n_particles, exp):

    particles = []
    weights = []
    logZs = []
    output = lambda x: x

    for i in range(n_particles):

        res = evaluate(exp, env=None)('addr_start', output)
        logW = 0.


        particles.append(res)
        weights.append(logW)

    #can't be done after the first step, under the address transform, so this should be fine:
    done = False
    smc_cnter = 0
    while not done:
        print('In SMC step {}, Zs: '.format(smc_cnter), logZs)
        for i in range(n_particles): #Even though this can be parallelized, we run it serially
            res = run_until_observe_or_end(particles[i])
            if 'done' in res[2]: #this checks if the calculation is done
                particles[i] = res[0]
                if i == 0:
                    done = True  #and enforces everything to be the same as the first particle
                    address = ''
                else:
                    if not done:
                        raise RuntimeError('Failed SMC, finished one calculation before the other')
            else:
                # pass #TODO: check particle addresses, and get weights and continuations
                particles[i] = res
                weights[i] = res[2]['logW']

        if not done:
            #resample and keep track of logZs
            logZn, particles = resample_particles(particles, weights)
            logZs.append(logZn)
        smc_cnter += 1
    logZ = sum(logZs)
    return logZ, particles


if __name__ == '__main__':

    for i in range(2,3):
        with open('programs/{}.json'.format(i),'r') as f:
            exp = json.load(f)
        print("Program:")
        print(i)
        n_particles = 1000 #TODO
        logZ, particles = SMC(n_particles, exp)

        print('logZ: ', logZ)

        values = torch.stack(particles)
        #TODO: some presentation of the results
        mean = torch.mean(values.float(),0)
        var = torch.var(values.float(),0)
        print("Mean:")
        print(mean)
        print("Variance:")
        print(var)

        f = open("data/dataP{}NP{}.txt".format(i,n_particles), "a")
        f.write('Mean: {}, Variance: {}, logZ: {}'.format(mean,var,logZ))
        f.close()

        if i < 3:
            h = plt.hist(values,bins=20)
            plt.title('Histogram for Program {} with {} Particles'.format(i,n_particles))
            plt.xlabel("Output Value")
            plt.ylabel("Number of Samples in Bin")
            plt.savefig('plots/P{}NP{}.png'.format(i,n_particles))
            plt.savefig('plots/P{}NP{}.pdf'.format(i,n_particles))
            plt.clf()

        if i == 3:
            A = np.zeros((n_particles,17))
            for i in range(0,n_particles):
                for j in range(0,17):
                    A[i,j] = values[i][j]

            h = np.zeros((3,17))
            for i in range(0,17):
                h[:,i] = np.histogram(A[:,i],bins=3,range=(0.0,2.0))[0]

            plt.imshow(h)
            plt.colorbar().set_label("Number of Samples in Bin")
            # plt.colorbar()
            plt.title('2D Histogram for Program 3 with {} Particles'.format(n_particles))
            plt.xlabel("Time Step")
            plt.ylabel("Latent State")
            plt.savefig('plots/P3NP{}.png'.format(n_particles))
            plt.savefig('plots/P3NP{}.pdf'.format(n_particles))
            plt.clf()
