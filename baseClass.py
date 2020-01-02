
import math
from heapq import nsmallest
import math
import array
import random

import matplotlib
from matplotlib import colors
import numpy as np
import numpy.linalg as npla
from numpy import ones,vstack
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
# %matplotlib inline

import pyvisgraph as pvg
import shapely.geometry.polygon as sgp
import shapely.geometry.linestring as sgls
import shapely.geometry.point as spt
import shapely.affinity as saf
from shapely import geometry


from Point import Point
from Rectangle import Rectangle

class baseParent:
    """This is the abstract parent class and contains all methods common between other object classes"""

    def __init__(self, bottom_left, top_right, name):
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.name = name
        self.rotation = 0
        self.redAreas = []

    def intersects(self, other):
        """Description : 
        Checks if two Rectangle objects insects or overlaps

        Parameters : 
        Other(rectangle) : Rectangle object to be compared to current rectangle object

        Returns:
        Returns True if intersection occurs else returns false

        """
        return not (self.top_right.x < other.bottom_left.x or 
                    self.bottom_left.x > other.top_right.x or 
                    self.top_right.y < other.bottom_left.y or 
                    self.bottom_left.y > other.top_right.y)

    def get_new_point(self):
        """Desription:
        Generate a random point

        Parameters : 
        None

        Returns:
        New point object created

        """
        return Point(random.randint(0, self.top_right.x), random.randint(0, self.top_right.y))

    def rotate(self, idx):
        """Description:
        Rotation around the center axis which takes an index and rotates object on that index

        Parameters:
        idx(int) = Index of object in rooms object list

        Returns:
        True if rotated else false


        """


        centr_point = self.center(idx)

        center_point_x = self.redAreas[idx]["furniture"].top_right.x/2
        center_point_y = self.redAreas[idx]["furniture"].top_right.y/2

        new_point_x = centr_point.x - center_point_y
        new_point_y = centr_point.y - center_point_x

        temp = Point(new_point_x, new_point_y)
        if (self.can_fit(temp, self.redAreas[idx]["furniture"], idx)):
            self.redAreas[idx]["point"].x = new_point_x
            self.redAreas[idx]["point"].y = new_point_y

            self.redAreas[idx]["furniture"].top_right.x, self.redAreas[idx]["furniture"].top_right.y = self.redAreas[
                idx]["furniture"].top_right.y, self.redAreas[idx]["furniture"].top_right.x


            self.rotation = (self.rotation + 1) % 4


            return True

        else:

            return False

    def can_fit(self, point, furniture, idx):
        """Desciption:
        Validate that if the new Furniture object can fitin the room without overlapping
        any other furniture object. Rectangle class is used in this function

        Paramters:
        point(Point) : New point object where furniture is to be placed
        furniture(Furniture) : Furniture object that is to be fitted in new place
        idx : Index of furniture in room objects list

        Returns:
        True if object can fit else false


        """
#         print(point.x + furniture.top_right.x , self.top_right.x , point.y+furniture.top_right.y , self.top_right.y)
        if point.x + furniture.top_right.x > self.top_right.x or point.y+furniture.top_right.y > self.top_right.y:
            return False
        for item in self.redAreas:
            if item["id"] is not idx:
                temp = Rectangle(item["point"], Point(
                    item["point"].x+item["furniture"].top_right.x, item["point"].y+item["furniture"].top_right.y))
                if temp.intersects(Rectangle(point, Point(point.x+furniture.top_right.x, point.y+furniture.top_right.y))):
                    return False
        return True

    def add_redArea(self, point, furniture, idx):
        """Description:
        Add the furniture object to redAreas so that no other object can be placed here


        Paramters:
        point(Point) : New point object where furniture is placed
        furniture(Furniture) : Furniture object that is to be added to redArea
        idx : Index of furniture in room objects list


        Returns:
        No returns. Adds new dictionary object to redAreas list having these new features


        """
        self.redAreas.append({
            "id": idx,
            "point": point,
            "furniture": furniture,
        })

    def fit(self, furniture):
        """Description:
        Generate Random population

        Paramters:
        furniture(Furniture) : Furniture object that is to be added in a room

        Returns:
        No returns. Adds new furniture object to room if it can be placed in a room

        """
        for idx, item in enumerate(furniture):
            iterations = 1500
            while iterations > 0:
                point_to_fit = self.get_new_point()
                if self.can_fit(point_to_fit, item, idx):
                    break
                iterations = iterations - 1
            if iterations > 0:
                self.add_redArea(point_to_fit, item, idx)

    def center(self, i):
        """Description:
        calculates the center of furniture object

        Parameters:
        i(int) = Index of furniture in room objects list


        Returns:
        Center point of given object

        """
        temp1 = self.redAreas[i]["point"].x + \
            self.redAreas[i]["furniture"].top_right.x/2
        temp2 = self.redAreas[i]["point"].y + \
            self.redAreas[i]["furniture"].top_right.y/2
        return Point(temp1, temp2)

    def distance(self, p1, p2):
        """Description:
        Calculates the distance between two points


        Parameters:
        p1(Point) : Starting point
        p2(Point) : Ending Point

        Returns:
        Distance between starting and ending point

        """
        return math.sqrt(((p1.x-p2.x)**2)+((p1.y-p2.y)**2))

    def find_closest_wall(self, itemx):
        """Description
        Find the closest wall and returns the appropriate integer
        1 for add x, 2 for add y, 3 for subtract x , 4 for subtract y
        -1 in case the Furniture object is not placed in the room

        Parameters:
        itemx(Furniture) = Furniture object whose closest wall is to be found

        Retruns:
        1(int) : If closest wall is Right in a rectangular box
        2(int) : If closest wall is Upper in a rectangular box
        3(int) : If closest wall is Left in a rectangular box
        4(int) : If closest wall is Bottom in a rectangular box

        """
        for item in self.redAreas:
            if item["furniture"] is itemx:
                x1 = item["point"].x
                y1 = item["point"].y
                x2 = x1 + item["furniture"].top_right.x
                y2 = y1 + item["furniture"].top_right.y

                min_dist = min(
                    [x1, y1, self.top_right.x-x2, self.top_right.y-y2])
                if min_dist == x1:
                    return 3
                elif min_dist == y1:
                    return 4
                elif min_dist == self.top_right.x-x2:
                    return 1
                elif min_dist == self.top_right.y-y2:
                    return 2
        return -1

    def find_closest_object(self, itemx):
        """Description:
        Finds the closest object and return s the index of that object and
        the index of the object passed to it in Room redAreas

        Parameters: 
        itemx(Furniture) = Furniture object whose closest object is to be found

        Returns:
        Index of current object
        Index of closest object found 

        """
        distances = []
        temp = Point(0, 0)
        for item in self.redAreas:
            if item["furniture"] is itemx:
                temp.x = int(item["point"].x +
                             item["furniture"].top_right.x / 2)
                temp.y = int(item["point"].y +
                             item["furniture"].top_right.y / 2)

        for item in self.redAreas:
            x1 = int(item["point"].x + item["furniture"].top_right.x / 2)
            y1 = int(item["point"].y + item["furniture"].top_right.y / 2)

            distances.append(self.distance(Point(x1, y1), temp))

        if distances:
            return (distances.index(nsmallest(2, distances)[-1]), distances.index(min(distances)))
        else:
            return (-1, -1)

    def move(self, itemx, direction):
        """Description:
        Snap the Furniture object to the closest wall


        Parameters:
        itemx(Furniture) = Furniture object that is to be moved
        Direction(int) = Direction in which object is to be moved

        Returns:
        None"""

#         all_objects_having_back = ['Chair']

#         for item in self.redAreas:
#             if item["furniture"] is itemx:
#                 can_snap = True

#                 if item["furniture"].additional_attr:
#                     if 'back' in item["furniture"].additional_attr['snap_direct']:
#                         if not((self.rotation + direction)%4  == 2):  
#                             can_snap = False


#                 if direction == 1 and can_snap:
#                     if self.can_fit(Point(self.top_right.x - item["point"].x - 1, item["point"].y), item["furniture"], item["id"]):
#                         item["point"].x = self.top_right.x - item["point"].x-1
#                         print("snapped")
#                     else:
#                         print("cannot fit")
#                 elif direction == 2 and can_snap:
#                     if self.can_fit(Point(item["point"].x, self.top_right.y - item["point"].y - 1), item["furniture"], item["id"]):
#                         item["point"].y = self.top_right.y - \
#                             item["point"].y - 1
#                         print("snapped")
#                     else:
#                         print("cannot fit")
#                 elif direction == 3 and can_snap:
#                     if self.can_fit(Point(1, item["point"].y), item["furniture"], item["id"]):
#                         item["point"].x = 1
#                         print("snapped")
#                     else:
#                         print("cannot fit")
#                 elif direction == 4 and can_snap:
#                     if self.can_fit(Point(item["point"].x, 1), item["furniture"], item["id"]):
#                         item["point"].y = 1
#                         print("snapped")
#                     else:
#                         print("cannot fit")

    def move_object(self, itemx, index):
        """Description:
        Snap the Furniture object to the closest object

        Parameters:
        itemx(Furniture) = Furniture object that is to be moved
        index(int) = Index of furniture in room objects list

        Returns :
        True if moved else false


        """
        point_list = []
        temp_x = self.redAreas[index[1]]["point"].x + \
            self.redAreas[index[1]]["furniture"].top_right.x + 1
        temp_y = self.redAreas[index[1]]["point"].y
        point_list.append(Point(temp_x, temp_y))
        temp_x = self.redAreas[index[1]]["point"].x
        temp_y = self.redAreas[index[1]]["point"].y + \
            self.redAreas[index[1]]["furniture"].top_right.y + 1
        point_list.append(Point(temp_x, temp_y))
        temp_x = self.redAreas[index[1]]["point"].x
        temp_y = self.redAreas[index[1]]["point"].y - \
            self.redAreas[index[0]]["furniture"].top_right.y - 1
        point_list.append(Point(temp_x, temp_y))
        temp_x = self.redAreas[index[1]]["point"].x - \
            self.redAreas[index[0]]["furniture"].top_right.x - 1
        temp_y = self.redAreas[index[1]]["point"].y
        point_list.append(Point(temp_x, temp_y))

        temp_x = self.redAreas[index[0]]["point"].x + \
            self.redAreas[index[0]]["furniture"].top_right.x/2
        temp_y = self.redAreas[index[0]]["point"].y + \
            self.redAreas[index[0]]["furniture"].top_right.y/2
        temp_point = Point(temp_x, temp_y)

        distances = []
        for item in point_list:
            distances.append(self.distance(temp_point, item))

        new_point = point_list[distances.index(min(distances))]
        if self.can_fit(new_point, self.redAreas[index[0]]["furniture"], self.redAreas[index[0]]["id"]):
            self.redAreas[index[0]]["point"] = new_point
            return True
        return False

    def align_it(self, itemx, direction):
        """Description:
        Align the Furniture object to the closest wall

        Parameters:
        itemx(Furniture) = Furniture object that is to be moved
        index(int) = Index of furniture in room objects list


        Returns :
        True if alligned else false


        """
        for item in self.redAreas:
            if item["furniture"] is itemx:
                if direction == 1:
                    new_y = self.top_right.y/2 - \
                        item["furniture"].top_right.y/2
                    if self.can_fit(Point(item["point"].x, new_y), item["furniture"], item["id"]):
                        item["point"].y = new_y
                        return True
                elif direction == 2:
                    new_x = self.top_right.x/2 - \
                        item["furniture"].top_right.x/2
                    if self.can_fit(Point(new_x - item["point"].y), item["furniture"], item["id"]):
                        item["point"].x = new_x
                        return True
                elif direction == 3:
                    new_y = self.top_right.y/2 - \
                        item["furniture"].top_right.y/2
                    if self.can_fit(Point(item["point"].x, new_y), item["furniture"], item["id"]):
                        item["point"].y = new_y
                        return True
                elif direction == 4:
                    new_x = self.top_right.x/2 - \
                        item["furniture"].top_right.x/2
                    if self.can_fit(Point(new_x, item["point"].y), item["furniture"], item["id"]):
                        item["point"].x = new_x
                        return True
                else:
                    return False

            return False

    def align_it_object(self, itemx, index):
        """Description:
        Align the Furniture object to the closest object

        Parameters:
        itemx(Furniture) = Furniture object that is to be moved
        index(int) = Index of furniture in room objects list


        Returns :
        True if alligned else false

        """
        item = self.redAreas[index[0]]
        new_y = item["point"].y + \
            item["furniture"].top_right.y/2 - itemx.top_right.y/2
        if self.can_fit(Point(item["point"].x, new_y), item["furniture"], item["id"]):
            self.redAreas[index[1]]["point"].y = new_y
            return True
        else:
            new_x = item["point"].x + \
                item["furniture"].top_right.x/2 - itemx.top_right.x/2
            if self.can_fit(Point(new_x - item["point"].y), item["furniture"], item["id"]):
                self.redAreas[index[1]]["point"].x = new_x
                return True
        return False

    def snap(self, item, thing):
        """Description:
        Control Flow of snap function

        Parameters:
        item(Furniture) = Furniture object that is to be moved
        thing = Wall or object 

        Return:
        True if moved else false

        """
        if thing == "wall":
            return(self.move(item, self.find_closest_wall(item)))
        elif thing == "object":
            return(self.move_object(item, self.find_closest_object(item)))
        else:
            return False

    def align(self, item, thing):
        """Description:
        Control Flow of align function

        Parameters:
        itemx(Furniture) = Furniture object that is to be moved
        thing = Wall or object 

        Return:
        True if aligned else false
        """
        if thing == "wall":
            return (self.align_it(item, self.find_closest_wall(item)))
        elif thing == "object":
            return(self.align_it_object(item, self.find_closest_object(item)))
        else:
            return False

    def pos_change(self ,item_index, distance ):
        """Description:
        Changes the position of the object according to distance defined

        Parameters:
        index(int) = Index of furniture in room objects list
        distance(int) = Sigma to gaussian distribution

        Returns:
        None
        """
        centr = self.center(item_index)

        all_points = np.random.normal(0, distance, 100)

        nmbr = random.randint(0,99)
        temp_x = self.redAreas[item_index]["point"].x + all_points[nmbr]
        temp_y = self.redAreas[item_index]["point"].y + all_points[nmbr]

        new_pos = Point( temp_x, temp_y) 
        if (self.can_fit(new_pos, self.redAreas[item_index]["furniture"], item_index)):

            self.redAreas[item_index]["point"].x = temp_x
            self.redAreas[item_index]["point"].y = temp_y

    def midpoint(self ,p1, p2):
        """Returns list of two midpoints"""
        return [(p1.x+p2.x)/2, (p1.y+p2.y)/2]

    def cost_function(self , objects_list ,roomPoints ,dR_room):



        clearance_proportion=self.clearnace_transformation(objects_list,roomPoints)

        circulation = self.circulation_transformation(objects_list)
        groupRelationship_rythm_goldSec = self.group_relationship_transformation(objects_list,roomPoints,
                                                                               dR_room )
        alignment = self.Aligment_transformation(objects_list , dR_room)

        actual_amounts = {'Table':2,'Chair':2,'Bed':2}
        desired_amounts = {'Table':1,'Chair':4,'Bed':1}
        Functionality = self.calcFunctionality(self.object_importance,actual_amounts
                                               ,desired_amounts)*3
        result = clearance_proportion + circulation + groupRelationship_rythm_goldSec + Functionality + alignment
        return(result)

    def clearnace_transformation (self , objects_list,roomPoints):
        """
        Transformation function for clearance and proportion
                """
        all_objects_points = []

        for item in objects_list:
            current_object_points = []
            x1 =int(item["point"].x)
            y1 = int(item["point"].y)
            x2 = x1 + int(item["furniture"].top_right.x)
            y2 = y1 + int(item["furniture"].top_right.y)
            p1 = geometry.Point(x1,y1)
            p2 = geometry.Point(x2,y1)
            p3 = geometry.Point(x2,y2)
            p4 = geometry.Point(x1,y2)
            current_object_points = [p1,p2,p3,p4]
            all_objects_points.append(current_object_points)

        clearance = self.calcLayoutClearance(all_objects_points)

#         roomPoints= [geometry.Point(0,0),geometry.Point(100,0)
#                      ,geometry.Point(100,100),geometry.Point(0,100)]

        proportion = self.calcProportion(all_objects_points,roomPoints)*2.5

        return (clearance + proportion)

    def circulation_transformation (self ,objects_list):

        """
        Transforms 
        """

        all_objects_points = []
        sp_list = []
        tp_list = []
        for item in objects_list:
            current_object_points = []
            x1 =int(item["point"].x)
            y1 = int(item["point"].y)
            x2 = x1 + int(item["furniture"].top_right.x)
            y2 = y1 + int(item["furniture"].top_right.y)
            p1 = geometry.Point(x1,y1)
            p2 = geometry.Point(x2,y1)
            p3 = geometry.Point(x2,y2)
            p4 = geometry.Point(x1,y2)

            sp = geometry.Point(0,0)
            tp = p1

            current_object_points = [p1,p2,p3,p4]
            all_objects_points.append(current_object_points)
            sp_list.append(sp)
            tp_list.append(tp)
        cost = self.calcLayoutCirculation(all_objects_points,sp_list,tp_list)*1.1
        return(cost)

    def group_relationship_transformation(self ,objects_list ,roomPoints , dR_room):
        """
        Transformation function for group_relationship and rythm and GoldenSec 
                """
        all_items_center = []
        furniture_type = []
        for idx,item in enumerate(objects_list):
            centr_point = baseParent.center(self , idx)
            all_items_center.append(geometry.Point(centr_point.x,centr_point.y))

            furniture_type.append(item["furniture"].room_type)

#         roomPoints= [geometry.Point(0,0),geometry.Point(100,0)
#                      ,geometry.Point(100,100),geometry.Point(0,100)]

#         dR_room = np.sqrt(100**2+100**2)  #Diagonal Size of the room
        group_Relation = self.calcGroupRelation(all_items_center,furniture_type,dR_room)
        objectDistribution = self.calcObjDistrib(all_items_center)
        goldenSection = self.calcGoldenSec(all_items_center,roomPoints,dR_room)*0.5

        return(group_Relation + objectDistribution + goldenSection)

    def Aligment_transformation(self , objects_list , dR_room):

        all_objects_front_center_points = []
        all_objects_back_center_points = []
        for item in objects_list:
            x1 =int(item["point"].x)
            y1 = int(item["point"].y)
            x2 = x1 + int(item["furniture"].top_right.x)
            y2 = y1 + int(item["furniture"].top_right.y)
            if (self.rotation == 0):
                front = self.midpoint(Point(x1,y1),Point(x2,y1))
                back = self.midpoint(Point(x1,y2),Point(x2,y2))
            elif (self.rotation == 1):
                front = self.midpoint(Point(x1,y2),Point(x1,y1))
                back = self.midpoint(Point(x2,y2),Point(x2,y1))
            elif (self.rotation == 2):
                front = self.midpoint(Point(x2,y2),Point(x1,y2))
                back = self.midpoint(Point(x2,y1),Point(x1,y1))
            elif (self.rotation == 3):
                front = self.midpoint(Point(x2,y1),Point(x2,y2))
                back = self.midpoint(Point(x1,y1),Point(x1,y2))
            all_objects_front_center_points.append(front)
            all_objects_back_center_points.append(back)
#         all_objects_front_points.append(front)
        walls = []
        walls.append(((0, 0), (100, 0)))
        walls.append(((100, 0), (100, 100)))
        walls.append(((100, 100), (0, 100)))
        walls.append(((0, 100), (0, 0)))


        wallProbVec = np.array([0.2, 0.2, 0.4, 0.6,0.2,0.6])
#         dR = 600

        return (self.calcAlignment(all_objects_back_center_points,walls,wallProbVec,dR_room))

    def calcLayoutClearance(self , objList, layoutPoly= None, entList = None):
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
            objListSp.append(sgp.Polygon([[p.x, p.y] for p in objList[n]]))

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

    def findPathPoly(self ,sourceP, targetP, objList, layoutPoly):

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
            for p in objList[n]:
                tmpPoly.append(pvg.Point(p.x,p.y))
            objListVg.append(tmpPoly)

        # Start building the visibility graph
        graph = pvg.VisGraph()
        refPoint = pvg.Point(sourceP[0].x, sourceP[0].y)
        workers = 1
        graph.build_mod(objListVg, workers, None, refPoint)  # , workers=workers)
#         graph.build(objListVg) #, workers=workers)

        # Get the shortest path
        shortest_path = []
        path_distance = []
        direct_distance = []

        for n in range(len(sourceP)):
            sP = pvg.Point(sourceP[n].x, sourceP[n].y)
            tP = pvg.Point(targetP[n].x, targetP[n].y)
            spath = graph.shortest_path(sP, tP)

            # Calculate the total distance of the shortest path
            pdistance = 0
            prev_point = spath[0]
            for point in spath[1:]:
                pdistance += np.sqrt((prev_point.y - point.y) ** 2 + (prev_point.x - point.x) ** 2)
                prev_point = point

            shortest_path.append(spath)
            path_distance.append(pdistance)
            dDist = np.sqrt((targetP[n].x - sourceP[n].x) ** 2 + (targetP[n].y - sourceP[n].y) ** 2)
            direct_distance.append(dDist)
        # print('Shortest path distance: {}'.format(path_distance))

        return shortest_path, path_distance, direct_distance

    def calcLayoutCirculation(self ,objList, srcList, tgtList):

        """
        calculating layout polygons accessibility from entrance (could be more than one entrance)  
        objList - List of obstacle objects (polygons)
                    Each object is assumed to represent the EXTENDED-bounding-box, i.e., including the
                    extra gap
                    required around the object 
        src/tgt-List - pairs of points between which shortest path is calculated and compared to straight
        path 
        """
#         print(objList)
#         print(srcList)
        nPairs = len(srcList)
        pathRatioSum = 0

        sPath, lenPath, dirPath = self.findPathPoly(srcList, tgtList, objList, [])

        for n in range(nPairs):
            pathRatioSum += (1 - dirPath[n] / lenPath[n])

        return pathRatioSum

    def calcGroupRelation(self ,objPos, membershipVec, dR):

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

    def calcObjDistrib(self ,objPos):

        """
        calculating object distribution in space, also referred to as Rhythm 
        """

        nObj = len(objPos)

        # get all pairs distance
        dP = np.array([])
        for i in range(nObj - 1):
            for j in range(i + 1, nObj):
                dP = np.append(dP, npla.norm(np.array(objPos[i]) 
                                             - np.array(objPos[j])))

        dMx = np.max(dP)
        dP /= dMx
        dPmean = np.median(dP)

        dSum = 0
        for n in range(len(dP)):
            dSum += (dP[n] - dPmean) ** 2
        dSum /= len(dP)  # ((nObj-1)*nObj/2)

        #  ==>>>>  Maybe calculate sqrt(dSum), i.e. the Sigma and not Variance

        return dSum

    def calcFunctionality(self,impVec, objCatNum, catDesNum):
        """
        calculating objects functionality importance and quantity 
         impVec - vector of importance values
         objCatNum - amount of objects from each category in the layout (dict)
         catDesNum - desired amount of objects from each category (dict)  

         #CALLING
         actual_amounts = {'Table':2,'Chair':2,'Bed':2}
        desired_amounts = {'Table':1,'Chair':4,'Bed':1}

        r1.calcFunctionality(r1.object_importance,actual_amounts,desired_amounts)
        """

        nO = len(impVec)

        fSum1 = np.sum(1-impVec)
        fSum1 /= nO

        fSum2 = 0
        for oc in objCatNum.keys():
            fSum2 += abs(objCatNum[oc] - catDesNum[oc])
        fSum2 /= (1.0 * len(objCatNum))

        fSum = 0.5 * (fSum1 + fSum2)
        return fSum

    def calcProportion(self ,objList, roomPoints, desRatio = 0.45):
        """

        Till now on the basis of area not volume

        calculating layout-volume-to-room-ratio
         objList: List of all points of each object in room
         roomVol: Room points
        """

        nObj = len(objList)
        objListSp = []
        # Transform to shapely
        for n in range(nObj):
            objListSp.append(sgp.Polygon([[p.x, p.y] for p in objList[n]]))

        roomSp = sgp.Polygon([p.x, p.y] for p in roomPoints)
        objVolSum = 0
        for i in range(len(objListSp)):
            objVolSum += objListSp[i].area

        roomVol = roomSp.area

        gP = max(desRatio - 1.0 * objVolSum / roomVol, 0) / (1.0 * desRatio)
        return gP

    def calcGoldenSec(self,objPos, roomRect, dR):
        """
        calculating objects location w.r.t. golden section lines
         objPos: objects' center position
         roomRect: 4 points of room (or sub-area) rectangle  
         dR: room diagonal
        """

        # make sure the vertices are ordered
        tmpRect = sgp.Polygon([p.x, p.y] for p in roomRect)
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

    def calcAlignment(self,backPos, walls, wallProbVec, dR):
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
    
    
        
        
        
class Room(baseParent):

    """Class describing room attributes"""
    def __init__(self, bottom_left, top_right, name,object_importance,doors=None
                 , windows=None ):
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.name = name
        self.doors = doors
        self.windows = windows
        self.object_importance = object_importance
        self.rotation = 0
        self.redAreas = []


    def get_new_point(self):
        """Description:
        Generate a random point

        Parameters:
        None

        Returns:
        None

        """
        return Point(random.randint(0, self.top_right.x), random.randint(0, self.top_right.y))

    def rotate(self, idx):
        """Rotation around the axis which takes an index and rotates object on that index"""

        """Description:
        Rotation around the center axis which takes an index and rotates object on that index

        Parameters:
        idx(int) = Index of object in rooms object list

        Returns:
        True if rotated else false


        """
        centr_point = self.center(idx)

        center_point_x = self.redAreas[idx]["furniture"].top_right.x/2
        center_point_y = self.redAreas[idx]["furniture"].top_right.y/2

        new_point_x = centr_point.x - center_point_y
        new_point_y = centr_point.y - center_point_x

        temp = Point(new_point_x, new_point_y)
        if (self.can_fit(temp, self.redAreas[idx]["furniture"], idx)):
            self.redAreas[idx]["point"].x = new_point_x
            self.redAreas[idx]["point"].y = new_point_y

            self.redAreas[idx]["furniture"].top_right.x, self.redAreas[idx]["furniture"].top_right.y = self.redAreas[
                idx]["furniture"].top_right.y, self.redAreas[idx]["furniture"].top_right.x

            self.rotation = (self.rotation + 1) % 4
            return True

        else:
            return False

    def move(self, itemx, direction):
        """Description:
        Snap the Furniture object to the closest object

        Parameters:
        itemx(Furniture) = Furniture object that is to be moved
        index(int) = Index of furniture in room objects list

        Returns :
        True if moved else False


        """

        all_objects_having_back = ['Chair']

        count = 0
        for item in self.redAreas:

            count +=1
            if item["furniture"] is itemx:

                can_snap = True
                if item["furniture"].additional_attr:
                    if 'back' in item["furniture"].additional_attr['snap_direct']:
                        if not((self.rotation + direction)%4  == 2):
                            can_snap = False

                            return False

                if direction == 1 and can_snap:
                    if self.can_fit(Point(self.top_right.x - item["point"].x - 1, item["point"].y), item["furniture"], item["id"]):
                        item["point"].x = self.top_right.x - item["point"].x-1

                        return True
                    else:

                        return False
                elif direction == 2 and can_snap:
                    if self.can_fit(Point(item["point"].x, self.top_right.y - item["point"].y - 1), item["furniture"], item["id"]):
                        item["point"].y = self.top_right.y - \
                            item["point"].y - 1

                        return True
                    else:

                        return False
                    return True
                elif direction == 3 and can_snap:
                    if self.can_fit(Point(1, item["point"].y), item["furniture"], item["id"]):
                        item["point"].x = 1

                        return True
                    else:

                        return False

                elif direction == 4 and can_snap:
                    if self.can_fit(Point(item["point"].x, 1), item["furniture"], item["id"]):
                        item["point"].y = 1

                        return True
                    else:

                        return False

    def squeeze(self):

        for count in range(len(self.redAreas)):
            self.move_object(self.redAreas[count]['furniture'], (count,0))

    def spread(self):

        for count in range(len(self.redAreas)):
            self.align(self.redAreas[count]['furniture'],"wall")
            self.snap(self.redAreas[count]['furniture'],"wall")
    def get_line_equation(self ,start, end):
    
        points = [start,end]
        x_coords, y_coords = zip(*points)
        A = vstack([x_coords,ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords)[0]
        
        return m , c
        print("Line Solution is y = {m}x + {c}".format(m=m,c=c))
    
    def blocker(self , src_point , target_point):
        
        m , c = self.get_line_equation(src_point , target_point)
        
        point_x = random.randint(src_point[0] , target_point[0])
        
        point_y = m*(point_x) + c
        
        self.redAreas[0]["point"].x , self.redAreas[0]["point"].y= point_x , point_y



class Furniture():
    """Class describing Furniture attributes

        Parameters:
        bottom_left = starting point of furniture object
        top_right = Top right corner points of furniture object
        name = Name of furniture object
        room_type = Table:1 | Chair:2 | Bed:3 """

    def __init__(self, bottom_left, top_right, name,room_type,parent=None, childs=None, additional_attr=None):
        self.bottom_left = bottom_left
        self.top_right = top_right
        self.name = name
        self.parent = parent
        self.childs = childs
        self.room_type = room_type
        self.additional_attr = additional_attr
        self.redAreas = []
