def filter(rfs, f):
    ro = rfs
    rfs = ro + f - 1
    print("receptive_field_size=",ro,"+",f,"-",1,"=",rfs)
    return rfs

def pool(rfs, S):
    ro = rfs
    rfs = (ro * S) - (S-1)
    print("receptive_field_size=",ro,"x",S, "-", S,"-",1,"=",rfs)
    return rfs

def get_rfs_without_bottleneck(rfs=1):

    print("\nResNet without bottleneck")
    print("Each block: 3x3 -> 3x3")

    print("\nreceptive_field_size=", rfs)

    print("\nFilter")
    for i in range(3):
        rfs = filter(rfs, 3)
        rfs = filter(rfs,3)
    print("Pool")
    rfs = pool(rfs, 2)

    print("\nFilter")
    for i in range(6):
        rfs = filter(rfs,3)
        rfs = filter(rfs,3)
    print("Pool")
    rfs = pool(rfs, 2)

    print("\nFilter")
    for i in range(4):
        rfs = filter(rfs,3)
        rfs = filter(rfs,3)
    print("Pool")
    rfs = pool(rfs, 2)

    print("\nFilter")
    for i in range(3):
        rfs = filter(rfs,3)
        rfs = filter(rfs,3)

    print("\nTAIL")
    print("\nPool")
    rfs = pool(rfs, 2)
    rfs = pool(rfs, 2)
    print("Filter")
    rfs = filter(rfs,7)

    print("\nreceptive_field_size=",rfs)

def get_rfs_with_bottleneck(rfs=1):

    print("\nResNet with bottleneck")
    print("Each block: 1x1 -> 3x3 -> 1x1")

    print("\nreceptive_field_size=", rfs)

    print("\nFilter")
    for i in range(3):
        rfs = filter(rfs, 3)
    print("Pool")
    rfs = pool(rfs, 2)

    print("\nFilter")
    for i in range(6):
        rfs = filter(rfs, 3)
    print("Pool")
    rfs = pool(rfs, 2)

    print("\nFilter")
    for i in range(4):
        rfs = filter(rfs, 3)
    print("Pool")
    rfs = pool(rfs, 2)

    print("\nFilter")
    for i in range(3):
        rfs = filter(rfs, 3)

    print("\nTAIL")
    print("\nPool")
    rfs = pool(rfs, 2)
    rfs = pool(rfs, 2)
    print("Filter")
    rfs = filter(rfs, 7)

    print("\nreceptive_field_size=", rfs)

rfs = 1

get_rfs_without_bottleneck(rfs)
get_rfs_with_bottleneck(rfs)