from itertools import accumulate
l=[8,2,3,4,5,6,7,8,9,10]
x=accumulate(l,max)
print([i for i in x])