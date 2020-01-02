"""
"""

# -----------------------------------------------------------------------------------------
import numpy as np
import numpy.linalg as npla
# ==> should install pyvisgraph
import pyvisgraph as pvg
# import scipy as sp
import shapely.geometry.polygon as sgp
import shapely.geometry.linestring as sgls
import shapely.geometry.point as spt
import shapely.affinity as saf


def findPathPoly(sourceP, targetP, objList, layoutPoly):
    """
    calculating shortest path from sourceP point to targetP that avoid polygon shape obstacles
    sourceP/targetP - 2D points
    objList - List of obstacle objects (polygons, each is a list of 2D points). 
                Should Contains the object's polygon and forward facing edge ???
    layoutPoly - Nx2 list of ordered vertices defining a 2D polygon of N vertices - room polygon layout
                last point NEQ first point
    =>>>>>>> Assuming polygons DO NOT intersect  
    """

    nObj = len(objList)
    objListVg = []
    # Transform to pyvisgraph format
    for n in range(nObj):
        tmpPoly = []
        for k in range(len(objList[n])):
            tmpPoly.append(pvg.Point(objList[n][k][0], objList[n][k][1]))
        objListVg.append(tmpPoly)

    # Start building the visibility graph
    graph = pvg.VisGraph()
    refPoint = pvg.Point(sourceP[0][0], sourceP[0][1])
    workers = 1
    graph.build_mod(objListVg, workers, None, refPoint)  # , workers=workers)
    # graph.build(objListVg) #, workers=workers)

    # Get the shortest path
    shortest_path = []
    path_distance = []
    direct_distance = []

    for n in range(len(sourceP)):
        sP = pvg.Point(sourceP[n][0], sourceP[n][1])
        tP = pvg.Point(targetP[n][0], targetP[n][1])
        spath = graph.shortest_path(sP, tP)

        # Calculate the total distance of the shortest path
        pdistance = 0
        prev_point = spath[0]
        for point in spath[1:]:
            pdistance += np.sqrt((prev_point.y - point.y) ** 2 + (prev_point.x - point.x) ** 2)
            prev_point = point

        shortest_path.append(spath)
        path_distance.append(pdistance)
        dDist = np.sqrt((targetP[n][0] - sourceP[n][0]) ** 2 + (targetP[n][1] - sourceP[n][1]) ** 2)
        direct_distance.append(dDist)
    # print('Shortest path distance: {}'.format(path_distance))

    return shortest_path, path_distance, direct_distance


# ------------------------------------------------------------------------------------------

def calcLayoutClearance(objList, layoutPoly, entList):
    """
    calculating layout polygons mean overlap 
    objList - List of obstacle objects (polygons)
                Each object is assumed to represent the EXTENDED-bounding-box, i.e., including the extra gap
                required around the object 
    layoutPoly - Nx2 list of ordered vertices defining a 2D polygon of N vertices - room polygon layout
               last point NEQ first point
    entList - List of entrance line segments (2D points). Entrances should not be occluded
    """

    #
    #  =>>>>> CURRENTLY constraints are not included, e.g. entrance, window, power-outlet, TV
    #

    nObj = len(objList)
    objListSp = []
    # Transform to shapely
    for n in range(nObj):
        objListSp.append(sgp.Polygon(objList[n]))

    ovlpSum = 0
    for m in range(nObj - 1):
        for n in range(nObj):
            if m == n:
                continue
            ovlp = objListSp[m].intersection(objListSp[n]).area
            ovlpSum += ovlp / objListSp[m].area

    ovlpSum = ovlpSum / (nObj * (nObj - 1))

    # ==> entrance overlap
    # if entLine.touches(tmpPolyLayout) or entLine.intersects(tmpPolyLayout):
    #    ovlp = entLine.intersection(tmpPolyLayout).length / entLine.length

    return ovlpSum


# ------------------------------------------------------------------------------------------

def calcLayoutCirculation(objList, srcList, tgtList):
    """
    calculating layout polygons accessibility from entrance (could be more than one entrance)  
    objList - List of obstacle objects (polygons)
                Each object is assumed to represent the EXTENDED-bounding-box, i.e., including the extra gap
                required around the object 
    src/tgt-List - pairs of points between which shortest path is calculated and compared to straight path 
    """

    nPairs = len(srcList)
    pathRatioSum = 0

    sPath, lenPath, dirPath = findPathPoly(srcList, tgtList, objList, [])

    for n in range(nPairs):
        pathRatioSum += (1 - dirPath[n] / lenPath[n])

    return pathRatioSum


# ------------------------------------------------------------------------------------------

def calcGroupRelation(objPos, membershipVec, dR):
    """
    calculating object inter-group-relations: spread of objects from a group relative to space diagonal  
    objPos - vector of objects' center (numpy array)
    membershipVec - vector of objects' membership association (integers)
    dR - space diagonal (scalar) 
    """

    gSum = 0
    nObj = len(objPos)

    for i in range(nObj - 1):
        for j in range(i + 1, nObj):
            gSum += 1.0 * (not (membershipVec[i] - membershipVec[j])) * npla.norm(np.array(objPos[i]) - np.array(objPos[j]))

    gSum /= ((nObj - 1) * nObj / 2 * dR)

    return gSum


# ------------------------------------------------------------------------------------------

#### Remaining ##
def calcAlignment(backPos, walls, wallProbVec, dR):
    """
    calculating object alignment, currently only w.r.t. supporting wall  
    backPos - vector of objects' back position (numpy array)
    walls - list of walls, each represented as a line (end points) 
    wallProbVec - probability vector of objects' to stand against the wall
    dR - space diagonal (scalar) 
    """

    #
    #  ====>  DIDNOT check direction, i.e. that object is parallel/perpendicular to wall
    #

    nW = len(walls)
    nO = len(backPos)
    wLines = []
    for iW in range(nW):
        wLines.append(sgls.LineString((walls[iW][0], walls[iW][1])))

    wSum = 0
    for iO in range(nO):
        dP = np.array([])
        for iW in range(nW):
            # shortest distance to wall
            dP = np.append(dP, wLines[iW].distance(spt.Point(backPos[iO])))
        wSum += wallProbVec[iO] * min(dP)

    wSum /= (nO * dR)
    return wSum


# ------------------------------------------------------------------------------------------

def calcObjDistrib(objPos):
    """
    calculating object distribution in space, also referred to as Rhythm 
    """

    nObj = len(objPos)

    # get all pairs distance
    dP = np.array([])
    for i in range(nObj - 1):
        for j in range(i + 1, nObj):
            dP = np.append(dP, npla.norm(objPos[i] - objPos[j]))
    dMx = np.max(dP)
    dP /= dMx
    dPmean = np.median(dP)

    dSum = 0
    for n in range(len(dP)):
        dSum += (dP[n] - dPmean) ** 2
    dSum /= len(dP)  # ((nObj-1)*nObj/2)

    #  ==>>>>  Maybe calculate sqrt(dSum), i.e. the Sigma and not Variance

    return dSum


# ------------------------------------------------------------------------------------------

def calcViewFrust(objPairsPos, objPairsDir):
    """
    calculating viewing frustum of some pairs of objects, i.e., those objects must "see" each other
    Should take into account direction of objects (facing each other)
    """


# ------------------------------------------------------------------------------------------

def calcFunctionality(impVec, objCatNum, catDesNum):
    """
    calculating objects functionality importance and quantity 
     impVec - vector of importance values
     objCatNum - amount of objects from each category in the layout (dict)
     catDesNum - desired amount of objects from each category (dict)  
    """

    nO = len(impVec)

    fSum1 = np.sum(1. - impVec)
    fSum1 /= nO

    fSum2 = 0
    for oc in objCatNum.keys():
        fSum2 += abs(objCatNum[oc] - catDesNum[oc])
    fSum2 /= (1.0 * len(objCatNum))

    fSum = 0.5 * (fSum1 + fSum2)
    return fSum


# ------------------------------------------------------------------------------------------

def calcProportion(objVol, roomVol, desRatio):
    """
    calculating layout-volume-to-room-ratio
     objVol: array of objects' area 
     roomVol: room area
    """

    objVolSum = 0
    for i in range(len(objVol)):
        objVolSum += objVol[i]

    gP = max(desRatio - 1.0 * objVolSum / roomVol, 0) / (1.0 * desRatio)
    return gP


# ------------------------------------------------------------------------------------------

def calcGoldenSec(objPos, roomRect, dR):
    """
    calculating objects location w.r.t. golden section lines
     objPos: objects' center position
     roomRect: 4 points of room (or sub-area) rectangle  
     dR: room diagonal
    """

    # make sure the vertices are ordered
    tmpRect = sgp.Polygon(roomRect)
    tmpRect = tmpRect.convex_hull
    t_rect = tmpRect.exterior.coords[0:-1]

    # creating golden lines. Assuming gsRatio = 13/21
    # go over the 2 consecutive pair of vertices and generate the 4-lines, 2 in each side
    gsr = 13.0 / 21.0

    line1 = sgls.LineString((t_rect[0], t_rect[1]))
    length = npla.norm(np.array(t_rect[0]) - np.array(t_rect[1]))
    pt11 = line1.interpolate(length * (1.0 - gsr))
    pt12 = line1.interpolate(length * gsr)
    line3 = sgls.LineString((t_rect[2], t_rect[3]))
    length = npla.norm(np.array(t_rect[2]) - np.array(t_rect[3]))
    pt32 = line3.interpolate(length * (1.0 - gsr))
    pt31 = line3.interpolate(length * gsr)

    line2 = sgls.LineString((t_rect[1], t_rect[2]))
    length = npla.norm(np.array(t_rect[1]) - np.array(t_rect[2]))
    pt21 = line2.interpolate(length * (1.0 - gsr))
    pt22 = line2.interpolate(length * gsr)
    line4 = sgls.LineString((t_rect[3], t_rect[0]))
    length = npla.norm(np.array(t_rect[3]) - np.array(t_rect[0]))
    pt42 = line4.interpolate(length * (1.0 - gsr))
    pt41 = line4.interpolate(length * gsr)

    gsLines = []
    gsLines.append(sgls.LineString((pt11, pt31)))
    gsLines.append(sgls.LineString((pt12, pt32)))
    gsLines.append(sgls.LineString((pt21, pt41)))
    gsLines.append(sgls.LineString((pt22, pt42)))

    dObjGs = []
    for i in range(len(objPos)):
        dd = []
        for j in range(len(gsLines)):
            dd.append(gsLines[j].distance(spt.Point(objPos[i])))
        dObjGs.append(min(dd))

    gP = np.sum(dObjGs)
    gP /= (1.0 * dR * len(objPos))

    return gP
