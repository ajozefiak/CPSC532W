using LinearAlgebra


##first define the probability distributions as defined in the excercise:

#define 1 as false, 2 as true
function p_C(c)
    p = [0.5, 0.5]
    return p[c]
end


function p_S_given_C(s,c)
    p = [0.5 0.9; 0.5 0.1]
    return p[s,c]
end

function p_R_given_C(r,c)
    p = [0.8 0.2; 0.2 0.8]
    return p[r,c]
end

function p_W_given_S_R(w,s,r)
    p = [1.0 0.0 0.1 0.9 0.1 0.9 0.01 0.99]
    p = reshape(p,(2,2,2))
    return p[w,s,r]
end

##1. enumeration and conditioning:

## compute joint:
p = zeros((2,2,2,2)) #c,s,r,w
for c in 1:2
    for s in 1:2
        for r in 1:2
            for w in 1:2
                p[c,s,r,w] = p_C(c)*p_S_given_C(s,c)*p_R_given_C(r,c)*p_W_given_S_R(w,s,r)
            end
        end
    end
end

## condition and marginalize:
p_C_given_W = 0.0
p_C_and_W = 0.0
p_W =  0.0
for c in 1:2
    for s in 1:2
        for r in 1:2
            global p_W += p[c,s,r,2]
        end
    end
end
for s in 1:2
    for r in 1:2
        global p_C_and_W += p[2,s,r,2]
    end
end
p_C_given_W = p_C_and_W / p_W

println("There is a ", p_C_given_W, " chance it is cloudly given the grass is wet")

print('There is a {:.2f}% chance it is cloudy given the grass is wet'.format(p_C_given_W[1]*100))



##2. ancestral sampling and rejection:
num_samples = 10000
samples = zeros(num_samples)
rejections = 0
i = 1
while i <= num_samples
    C = 0
    S = 0
    R = 0
    W = 0
    p = rand()
    if p > p_C(1)
        C = 2
    else
        C = 1
    end
    p = rand()
    if p > p_S_given_C(1,C)
        S = 2
    else
        S = 1
    end
    p = rand()
    if p > p_R_given_C(1,C)
        R = 2
    else
        R = 1
    end
    p = rand()
    if p > p_W_given_S_R(1,S,R)
        W = 2
    else
        W = 1
    end
    if W == 2
        global samples[i] = C-1
        global i += 1
    else
        global rejections += 1
    end
end


print('The chance of it being cloudy given the grass is wet is {:.2f}%'.format(samples.mean()*100))
print('{:.2f}% of the total samples were rejected'.format(100*rejections/(samples.shape[0]+rejections)))


#3: Gibbs
# we can use the joint above to condition on the variables, to create the needed
# conditional distributions:


#we can calculate p(R|C,S,W) and p(S|C,R,W) from the joint, dividing by the right marginal distribution
#indexing is [c,s,r,w]
p_R_given_C_S_W = p/p.sum(axis=2, keepdims=True)
p_S_given_C_R_W = p/p.sum(axis=1, keepdims=True)


# but for C given R,S,W, there is a 0 in the joint (0/0), arising from p(W|S,R)
# but since p(W|S,R) does not depend on C, we can factor it out:
#p(C | R, S) = p(R,S,C)/(int_C (p(R,S,C)))

#first create p(R,S,C):
p_C_S_R = np.zeros((2,2,2)) #c,s,r
for c in range(2):
    for s in range(2):
        for r in range(2):
            p_C_S_R[c,s,r] = p_C(c)*p_S_given_C(s,c)*p_R_given_C(r,c)

#then create the conditional distribution:
p_C_given_S_R = p_C_S_R[:,:,:]/p_C_S_R[:,:,:].sum(axis=(0),keepdims=True)



##gibbs sampling
num_samples = 10000
samples = np.zeros(num_samples)
state = np.zeros(4,dtype='int')
#c,s,r,w, set w = True

#TODO

print('The chance of it being cloudy given the grass is wet is {:.2f}%'.format(samples.mean()*100))
