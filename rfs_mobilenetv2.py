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

def get_rfs(rfs, MODEL_LEVEL_2_BLOCKS, MODEL_LEVEL_1_BLOCKS, MODEL_LEVEL_0_BLOCKS):

    print("\nMobileNet V2 with bottle neck")
    print("Each block: 1x1 -> 3x3 -> 1x1")

    print("\nreceptive_field_size=", rfs)

    print("\nLevel2")
    print("\nFilter")
    for i in range(MODEL_LEVEL_2_BLOCKS):
        rfs = filter(rfs, 3)
    rfs = filter(rfs, 3)
    print("Pool")
    rfs = pool(rfs, 2)

    print("\nLevel1")
    print("\nFilter")
    for i in range(MODEL_LEVEL_1_BLOCKS):
        rfs = filter(rfs, 3)
    rfs = filter(rfs, 3)
    print("Pool")
    rfs = pool(rfs, 2)

    print("\nLevel0")
    print("\nFilter")
    for i in range(MODEL_LEVEL_0_BLOCKS):
        rfs = filter(rfs, 3)
    rfs = filter(rfs, 3)

    print("\nTAIL")
    print("Filter")
    rfs = filter(rfs, 3)

    print("\nreceptive_field_size=", rfs)

rfs = 1
MODEL_LEVEL_0_BLOCKS    = 4
MODEL_LEVEL_1_BLOCKS    = 6
MODEL_LEVEL_2_BLOCKS    = 3

get_rfs(rfs, MODEL_LEVEL_2_BLOCKS, MODEL_LEVEL_1_BLOCKS, MODEL_LEVEL_0_BLOCKS)