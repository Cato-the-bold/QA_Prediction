import bisect

def flowers(A, M, K):
    internals = [[0, len(A) - 1]]
    k = len(A)-1
    while k:
        pos = A[k]-1

        #find which internal to split.
        #version1
        # i = 0; j = len(internals)-1
        # while i<j:
        #     mid = (i+j)/2
        #     if internals[mid][0]>pos:
        #         j = mid-1
        #     elif internals[mid][1]<pos:
        #         i = mid+1

        #version2
        i = bisect.bisect(internals, [pos,float('inf')])-1

        internal = internals[i]
        if internal[0]==pos:
            internal[0] = internal[0]+1
        elif internals[i][1]==pos:
            internal[1] = internal[1]-1
        else:
            internals.insert(i+1,[pos+1,internal[1]])
            internal[1] = pos-1

        internals = [i for i in internals if i[1]-i[0]+1>=K]
        if len(internals)==M:
            return k

        k-=1

    return -1

print flowers([1,6,7,2,3,4,5], 2, 2)