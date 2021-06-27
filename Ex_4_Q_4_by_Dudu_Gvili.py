#A restaurant networks want to open branches along a highway
#there are 'n' available places given in list m[]
#at each location you can open 1 restaurant and the profit will be according to list p[] in same index of m[]
#the distance between 2 branches has to be atleast 'k'
#write an algorithm that find the maximum profit locations arangement/

#1. calculating list c[] of closest place to open a restaraunt if i opened one at m[i]
#2. making a list of T - total profit, R - opened restaurants
#3. comparing 2 options - not opening a branch at location i and then T[i] will be the same as T[i-1]
# or opening a branch at location i and taking that branch profit (p[i]) + the total profit of the branches before it in distance of atleast k (T[c[i]])


def maxProfitLocations(m, p, c):
    T = [0]
    R = [0]*len(m)ches 
    for i in range(1,n):
        open_i = T[i-1]
        no_open_i = p[i] + T[c[i]]
        if no_open_i > open_i:
            T[i] = no_open_i
        else:
            T[i] = open_i
            R[i] = 1
    return T, R

def findCi(m, k):
    c = [0]
    for i in range (1, len(m)):
        j = c[i-1]
        while (m[j+1] <= m[i]-k):
            j += 1
        c.append(j)
    return c

def driverCode(m, p, k):
    maxProfitLocations(m, p, findCi(m, k))

        
