
from WE.utils.data import DataLoader
from WE.utils.evaluate import *
from WE.extractor import Extractor
import time

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
                # fix!!!!!!
trainEpoch = 100

batchedTrainTree, batchTrainOutput = a.batchifyData(trainTree, trainOutput, batch_size = batchSize)
batchNum = len(batchedTrainTree)

### Extractor / Model parameters ###
model = 'modelTest'
extr = Extractor(model, debug = True)
    # TODO pass parameters.


#################################
##            Train            ##
#################################

debug = False
print("\n\n\n> Training " + '-'*50 )
startTime = time.time()

for epoch in range(trainEpoch):                              # iter EPOCH

    lastEpochTime = time.time()
    for batchId in range(batchNum) :                             # iter BATCH
        x_dom = batchedTrainTree[batchId]
        y_pairs_label = batchTrainOutput[batchId]

        # Train.
        y_pairs, modelAcc = extr.trainAndExtract(x_dom, y_pairs_label)
        P, R, Fscore = getExtractionPRF(y_pairs, y_pairs_label)
        if debug:
            print('>> Batch %3d / %3d | P : %1.3f , R : %1.3f , F-score : %1.3f'
                  % (batchId+1, batchNum, P, R, Fscore) )
        extr.optimize()

    # Evaluate.
    print("> EPOCH %3d  |  epochTime : %6.1fs  |  totalTime : %6.1fs"
          % (epoch+1, time.time() - lastEpochTime, time.time() - startTime) )
    # report epoch accuracy

    # TODO remove.
    break
print("\n\n\n> Test Result " + '-'*50 )

#test
# result, modelacc = Extractor.evaluate(html, output)
# get extractionaccuracy (output)


# report test accuracy
print("Total Time %6.1fs" % (time.time()-startTime) )

# TODO export / visualize data