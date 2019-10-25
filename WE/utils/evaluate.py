
### evaluate.py

def _regularizeString(string):
    return string.strip().strip(':').strip(';')

def _strcmp(str1: str, str2: str) -> bool:
    if str1.strip() == str2.strip(): return True
    else: return False

def _pairsIdentical(p1, p2):
    if _strcmp(p1[0], p2[0]) and _strcmp(p1[1], p2[1]): return True
    else: return False

def getExtractionPRF(outs, golds):

    TP = 0        # in both out, gold
    FN = 0        # in gold, not in out
    FP = 0        # in out , not in gold
                  # do not define TN

    for idx, out in enumerate(outs):
        for dp in out:
            correct = False
            for g in golds[idx]:
                if _pairsIdentical(dp, g):
                    correct = True
            if correct : TP += 1
            else: FP += 1

        for g in golds[idx]:
            found = False
            for dp in out:
                if _pairsIdentical(dp, g):
                    found = True
            if found: pass
            else : FN += 1

    P = TP / (TP+FP)
    R = TP / (TP+FN)
    F1 = 2 * (P*R) / (P+R)

    return P, R, F1