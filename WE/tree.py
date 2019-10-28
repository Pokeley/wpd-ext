
### tree.py

from os import listdir
from os.path import isfile, join
from lxml import html
from WE.utils.evaluate import _strcmp, _regularizeString

def _reductable(tree, childs, me):
    result = False
    ### CASE 1
    if tree.text_content().strip() == '':
        result = True
    ### CASE 2
    myStr = tree.text_content()
    for child in childs:
        if myStr == child.text_content():
            result = True
            break
    ### CASE 3
    if _isIgnorableTag(me):
        result = True
    ### CASE 4
    if _isLeafnode(me) and me.text_content() == None:
        result = True

    return result

def _isIgnorableTag(node):
    # TODO check removing tags below.
    """
    Remove following tags.
        Confirmed : <head> <title> <meta> <link> <script> <style> <noscript>
        not sure yet : <header> <footer>
    """
    if node.tag in ['head', 'title', 'meta', 'link', 'script', 'style', 'noscript',
                    'header', 'footer']:
        return True
    return False

def _shrinkDFS(tree, parent, level:int):
    parent = parent
    me = tree.xpath(".")[0]
    childs = tree.xpath("./*")

    # decision
    reductable = _reductable(tree, childs, me)

    newParent = me
    if reductable:
        for child in childs:
            parent.append(child)
        parent.remove(me)
        newParent = parent

    for child in childs:
        _shrinkDFS(child, newParent, level+1)
    return

def _shrinkTree(tree):
    _shrinkDFS(tree, None, 1)

def _leafDFS(tree, out, nodeToIgnore = None):
    me = tree.xpath('.')[0]
    if _isLeafnode(me):
        # leaf nodes in tree after shrink should always contain text
        assert me.text_content() != ''
        out.append(me)
    childs = tree.xpath("./*")
    for child in childs:
        if child is not nodeToIgnore:
            _leafDFS(child, out)
    return

def _findLeafnodes(tree):
    output = []
    _leafDFS(tree, output)
    return output

def _isLeafnode(tree):
    return True if len(tree.xpath("./*")) == 0 and tree.text != None\
                else False

def _neighborSearch(node, lookupHeight):

    """
    Sequence ordering algorithm : first node is our base, increase height till lH and find neighbors.
    """
    result = [node, ]
    nodeToIgnore = node
    for height in range(1, lookupHeight+1):
        try:
            subtreeRoot = node.xpath('.'+ height*'/..')[0]
            _leafDFS(subtreeRoot, result, nodeToIgnore)
        except Exception as e:
            break
        nodeToIgnore = subtreeRoot

    return result
def _generateSearchDict(label, attrAsKeyDict, valueAsKeyDict):

    for pair in label:
        at = pair[0]
        val = pair[1]

        if at not in attrAsKeyDict.keys():
            attrAsKeyDict[at] = [val, ]
        else:
            attrAsKeyDict[at].append(val)

        if val not in valueAsKeyDict.keys():
            valueAsKeyDict[val] = [at, ]
        else:
            valueAsKeyDict[val].append(at)

def _generateLabels(nodes, attrDict, valueDict):
    # TODO might change
    """
    One-hot encoding for Labeling Rules. ---
        0 : main is attribute
        1 : main is value
        2 : main is nothing
        3 : neighbor is value
        4 : neighbor is attribute
        5 : neighbor is nothing.
        (?) 6 : other value.
        (?) 7 : other attribute.
    """
    result = []
    mainNode = nodes[0]

    mainCase = None
    mainText = _regularizeString(mainNode.text_content())

    if mainText in attrDict.keys():
        mainCase = 0
    elif mainText in valueDict.keys():
        mainCase = 1
    else:   # fill rest to 5.
        mainCase = 2

    result.append(mainCase)
    existed = False
    for nei in nodes[1:]:
        if mainCase == 0:
            exist = False
            for val in attrDict[mainText]:
                if _strcmp(val, _regularizeString(nei.text_content())):
                    exist = True
            if exist and not existed:
                result.append(3)
                existed = True
            else:
                result.append(5)
        elif mainCase == 1:
            exist = False
            for attr in valueDict[mainText]:
                if attr =='[NONE]':
                    break
                if _strcmp(attr, _regularizeString(nei.text_content())):
                    exist = True
            if exist and not existed:
                result.append(4)
                existed = True
            else:
                result.append(5)
        else :
            result.append(5)

    return result

def sequenceGenerator(DOMs, labels, maxSubtreeHeight = 2):

    """
    we first assume that only leaf node can be selected by our model.
    """
    nodeSequence = []
    labelSequence = []

    # TODO
    # if subTreeHeight > treeheight:
    #     raise ValueError

    for idx, tree in enumerate(DOMs):
        label = labels[idx]

        _shrinkTree(tree)  # Tree object modified.
        leaves = _findLeafnodes(tree)

        attrAsKeyDict = dict()
        valueAsKeyDict = dict()
        _generateSearchDict(label, attrAsKeyDict, valueAsKeyDict)

        # Debug
        # for k in attrAsKeyDict.keys():
        #     print(k, attrAsKeyDict[k])
        # for k in valueAsKeyDict.keys():
        #     print(k, valueAsKeyDict[k])

        for leaf in leaves:
            neiNodes = _neighborSearch(leaf, maxSubtreeHeight)
            nodeSequence.append(neiNodes)

            neiLabels = _generateLabels(neiNodes, attrAsKeyDict, valueAsKeyDict)
            labelSequence.append(neiLabels)

            assert len(neiNodes) == len(neiLabels)

        #     print(leaf.text_content().strip(), neiLabels)
        #     print([node.text_content().strip() for node in neiNodes])
        #
        #     print('-'*100)
        #
        # exit(0)

    return (nodeSequence, labelSequence)


if __name__ == '__main__':

    """
    Only for development purpose, ignore code below.
    """

    folderPath = '../../wdc-dataset/fextraction-subset-450/'
    htmls = [folderPath + f for f in listdir(folderPath) if isfile(join(folderPath, f))]
    try:
        htmls.remove('../../wdc-dataset/fextraction-subset-450/.DS_Store')
    except:
        pass

    print(">>> {} htmls in folder.".format(len(htmls)))

    # Iterate per html.
    statsArr = []
    for ht in htmls:
        fp = open(ht, "r", encoding='utf-8')
        content = fp.read()
        rootDOM = html.fromstring(content)
        fp.close()



