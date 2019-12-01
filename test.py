
from WE.utils.data import DataLoader
from WE.utils.evaluate import *
from WE.extractor import Extractor
import time
from tensorboardX import SummaryWriter


summary = SummaryWriter('./tflog/')


htmlPath = './dataset/fextraction-subset-450/'
exportPath = './resultTmp/'

a = DataLoader(debug = True)
trainSet, testSet = a.loadfromFile(['./dataset/변환_dataset_tableonly/all_450_table.json',]
                                   , trainRatio=.8
                                   , splitDomain=True)

# get each set into html / label.
trainInput = a.extractInputHTML(trainSet)
trainOutput = a.extractOutputAVPair(trainSet)
testInput = a.extractInputHTML(testSet)
testOutput = a.extractOutputAVPair(testSet)

trainTree = a.loadDOMTree(trainInput, htmlPath, True)
testTree = a.loadDOMTree(testInput, htmlPath, True)

### Hyperparameters ###
batchSize = 1   # 1 : sg, -1 : full-batch
trainEpoch = 100

batchedTrainTree, batchTrainOutput = a.batchifyData(trainTree, trainOutput, batch_size = batchSize)
batchedTestTree, batchTestOutput = a.batchifyData(testTree, testOutput, batch_size = 1)

batchNum = len(batchedTrainTree)
testBatchNum = len(batchedTestTree)

### Extractor / Model parameters ###
model = 'modelTest'
extr = Extractor(model, debug = True)
    # TODO pass parameters.



debug = False
startTime = time.time()
print('>> Train started.')


for epoch in range(trainEpoch):                              # iter EPOCH

    lastEpochTime = time.time()

    #################################
    ##            Train            ##
    #################################

    agg_batchLoss = 0.0
    actualValidData = 0
    agg_rate = 0.0

    for batchId in range(batchNum) :                             # iter BATCH
        x_dom = batchedTrainTree[batchId]
        y_pairs_label = batchTrainOutput[batchId]

        # Train.
        y_pairs, batchLoss, rate = extr.trainAndExtract(x_dom, y_pairs_label)
        if batchLoss != -1:
            P, R, Fscore = getExtractionPRF(y_pairs, y_pairs_label)
            if debug:
                print('>> Batch %3d / %3d | Loss : %1.3f, P : %1.3f , R : %1.3f , F-score : %1.3f'
                      % (batchId + 1, batchNum, batchLoss, P, R, Fscore))

            agg_batchLoss += batchLoss
            agg_rate += rate
            actualValidData += 1
        else:
            if debug :
                print(">> Batch %3d skipped." % (batchId + 1))

    print("> EPOCH %3d  |  epochTime : %6.1fs  |  totalTime : %6.1fs"
          % (epoch+1, time.time() - lastEpochTime, time.time() - startTime) )
    print("> train_batchloss : %.4f" % (agg_batchLoss / actualValidData, ))
    print("> train_avgrate : %.4f" % ( agg_rate / actualValidData, ))

    summary.add_scalar('tr_ba_loss', agg_batchLoss / actualValidData, epoch+1 )
    summary.add_scalar('tr_av_rate', agg_rate / actualValidData, epoch + 1)

    # + report epoch acc(model) acc(real).

    #################################
    ##            Test             ##
    #################################

    test_agg_batchLoss = 0.0
    test_actualValidData = 0
    test_agg_rate = 0.0
    test_P, test_R, test_F = 0, 0, 0

    for batchId in range(testBatchNum):
        x_dom = batchedTestTree[batchId]
        y_pairs_label = batchTestOutput[batchId]

        y_pairs, batchLoss, rate = extr.trainAndExtract(x_dom, y_pairs_label)
        if batchLoss != -1:
            P, R, Fscore = getExtractionPRF(y_pairs, y_pairs_label)
            test_P += P
            test_R += R
            test_F += Fscore

            test_agg_batchLoss += batchLoss
            test_agg_rate += rate
            test_actualValidData += 1

    if test_actualValidData == 0:
        print('wrong test set')
        exit(0)

    print('> test_PRF : %.4f %.4f %.4f' % (test_P/test_actualValidData , test_R/test_actualValidData, test_F/test_actualValidData))
    print("> test__batchloss : %.4f" % (test_agg_batchLoss / test_actualValidData,))
    print("> test_avgrate : %.4f" % ( test_agg_rate / test_actualValidData,))
    summary.add_scalar('te_ba_loss', test_agg_batchLoss / test_actualValidData , epoch+1 )
    summary.add_scalar('te_av_rate', test_agg_rate / test_actualValidData, epoch + 1)

    # TODO
    # save average loss.
    # if acc got better, save parameters. (write saving time...)

    # TODO
    # save everything into csv (PRF, etc..)


print("Total Time %6.1fs" % (time.time()-startTime) )

# TODO export / visualize data