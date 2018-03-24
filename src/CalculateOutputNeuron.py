
for j in range (226, 300):

    imgsize = j
    filtersize = 3
    padding = 2
    stride = 1


    pool_filtersize = 2
    pool_stride = 2

    print "\nJ---->>>>>" +str(j)

    for i in range(6):
        print "I---->>>>>" + str(i)

        conv_output = ((imgsize - filtersize + (2*padding)) / stride ) + 1
        print "convoutput: "+str(conv_output)

        pool_output = ((conv_output - pool_filtersize) / pool_stride) + 1
        print "pooloutput: " + str(pool_output)

        imgsize = pool_output

        if(conv_output % 2 != 0 or pool_output % 2 != 0):
            break


    conv_output = ((imgsize - filtersize + 2*padding) / stride ) + 1



    print "IMG SIZE: "+str(conv_output)

