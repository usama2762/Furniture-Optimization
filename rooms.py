import math

import matplotlib
import numpy as np

# matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# import pyvisgraph as pvg
import layoutCostFunctions

class Furniture:
    """
    Furniture is a class for handing furnitures data.
    It contains the data of the furniture needed in order to choose and arrange the furniture in the space.
    """

    def __init__(self, name, length, width, **kwargs):
        self.name = name
        self.length = length
        self.width = width
        self.proportions = np.array([length, width])
        if 'guard_array' in kwargs:
            self.guard_array = np.array(kwargs['guard_array'])
        else:
            self.guard_array = np.array([5, 5])

    def __str__(self):
        return '%s   ' % self.name

    def __repr__(self):
        return '%s   ' % self.name


class GeoItems:
    """    GeoItems is a class that consists data of a group of geographic items.

    namely it contains a list of the items, its position and orientations.
    It is used to create furnitures arrangements or room areas arrangements.
    init function creates empty object, and you add items to the object using add_item method.


    -------- **Attributes** --------

    list : list
        a list of Geographic Items
    positions: list
        a list of the positions (x,y) of the items
    orientations: list
        a list of the angle of the items

    -------- **Methods** --------

    *add_item*
        adds new item to GeoItems
    *length*
        returns a vector with the object lengths
    *dist*
        calculates the distance between furnitures
    *item*
        Returns the index number of the item 'name'
    """

    def __init__(self, items=[], positions=[], orientations=[]):
        self.items = items
        self.positions = positions
        self.orientations = orientations

    def __str__(self):
        return '%s' % self.items

    def __repr__(self):
        return '%s' % self.items

    def add_item(self, item, position, orientation):
        """
        adds item to the GeoItems
        :param item: item object
        :param position: (x,y)
        :param orientation: angle

        """
        self.items.append(item)
        self.positions.append(position)
        self.orientations.append(orientation)

    def length(self, ind):
        """ lengths returns a vector with the object lengths"""

        if self.orientations[ind] == 0 or self.orientations[ind] == 180:
            r = np.array([self.items[ind].proportions[0], self.items[ind].proportions[1]])
        else:
            r = np.array([self.items[ind].proportions[1], self.items[ind].proportions[0]])
        return r

    def dist(self, ind1, ind2):
        """ dist calculates the distance between furniture ind1 and furniture ind2 from the list The distance is from
        the end of the furniture (and not the centers) """

        centers_distance = (np.array(self.positions[ind1]) - np.array(self.positions[ind2]))
        r1 = self.length(ind1)
        r2 = self.length(ind2)
        d = np.abs(centers_distance) - (r1 + r2) / 2
        print(d)
        return d

    def item(self, name):
        """
        Returns the index number of the item called name
        :param name: string
        :return: index
        """
        indxs = []
        ind = 0
        for item in self.items:
            if name == item.name:
                indxs.append(ind)
            ind += 1
        return indxs

    def plot(self):
        area_fig = plt.figure()
        ax1 = area_fig.add_subplot(111, aspect='equal')
        for i in range(len(self.items)):
            d = self.length(i)
            left_down_corner = np.array(self.positions[i]) - np.array(d / 2)
            ax1.add_patch(patches.Rectangle(left_down_corner, d[0], d[1]))
        plt.xlim(-100, 1000)
        plt.ylim(-100, 1000)
        plt.draw_if_interactive()
        plt.show()


class RoomArea:
    """
    RoomArea is a class that consists a design of a specific area.


     -------- Attributes --------

    name : string
        the name of the design
    type : string
        the type of area (for example dinning area)
    proportions : string
        the length and width of the area (l,w)
    furnitures : GeoItems class
        a list of furniture items
    """

    def __init__(self, area_type, name, proportions):
        self.name = name
        self.type = area_type
        self.proportions = np.array(proportions)
        self.furnitures = GeoItems([], [], [])

    def __str__(self):
        return 'area %s room, name %s' % (self.type, self.name)

    def __repr__(self):
        return 'area %s room, name %s' % (self.type, self.name)


class Room(GeoItems):
    """
    a Room object represents the room by breaking it into different areas. each area is stored in the items property with
    its position and orientation. The area is represented by the class RoomArea.

    """

    def __init__(self, items=[], positions=[], orientations=[]):
        super().__init__(items, positions, orientations)

    def area_names(self, indx):
        """
        Returns the names of all furnitures in area of #index
        :param indx:
        :return: names list
        """
        names = []
        for i in range(len(self.items[indx].furnitures.items)):
            names.append(self.items[indx].furnitures.items[i].name)
        return names

    def area_orientation(self, indx):
        """
        Returns the orientation of all furnitures in area of #index
        :param indx:
        :return: orientations
        """
        orientations = np.mod(self.orientations[indx] + np.array(self.items[indx].furnitures.orientations), 360)
        return orientations

    def area_pos(self, indx):
        """
        Returns the positions of all furnitures in area of #index
        :param indx:
        :return: pos_list
        """
        orientation = self.orientations[indx]
        pos_list = []
        for i in range(len(self.items[indx].furnitures.items)):
            position = self.items[indx].furnitures.positions[i]
            d = {0: position, 90: np.array([-position[1], position[0]]), 180: -np.array(position),
                 270: np.array([position[1], -position[0]])}
            if not math.isnan(position[0]):
                pos_list.append(d[orientation] + np.array(self.positions[indx]))
            else:
                pos_list.append(np.nan)
        return pos_list

    def area_dims(self, indx):
        """
        Returns the dimensions of all furnitures in area of #index
        :param indx:
        :return: dim_list
        """
        orientation = self.area_orientation(indx)
        dim_list = []
        for i in range(len(orientation)):
            proportion = self.items[indx].furnitures.items[i].proportions
            d = {0: proportion, 90: np.array([proportion[1], proportion[0]]), 180: proportion,
                 270: np.array([proportion[1], proportion[0]])}
            if not math.isnan(orientation[i]):
                dim_list.append(d[orientation[i]])
            else:
                dim_list.append(np.nan)
        return dim_list

    def plot(self):
        room_fig = plt.figure()
        ax1 = room_fig.add_subplot(111, aspect='equal')

        # if room has room polygon
        if hasattr(self, 'poly'):
            plt.plot(np.vstack([self.poly, self.poly[0]])[:, 0], np.vstack([self.poly, self.poly[0]])[:, 1])

        for i in range(len(self.items)):
            pos = self.area_pos(i)
            dims = self.area_dims(i)
            orientation = self.area_orientation(i)
            poly = self.area_poly(i)
            if len(poly) == 4:
                poly = np.vstack([poly, poly[0]])  # complete the polygon if isn't complete
            plt.plot(poly[:, 0], poly[:, 1])

            for j in range(len(pos)):
                left_down_corner = np.array(pos[j]) - np.array(dims[j] / 2)
                if type(left_down_corner) == np.ndarray:  # not nan
                    print("position ", pos[j], 'dims', dims[j], ',orientation ', orientation[j])
                    ax1.add_patch(patches.Rectangle(left_down_corner, dims[j][0], dims[j][1]))
        # plt.xlim(-1000,1000)
        # plt.ylim(-1000,1000)
        plt.draw()
        # plt.savefig('room_image.png')
        plt.show()

    def area_poly(self, indx, add_guard=False):
        """
        Returns the polynom of area of #index
        :param indx:
        :return: array of area polynom
        """

        orientation = self.orientations[indx]
        proportions = self.items[indx].proportions
        if add_guard is True:
            proportions += self.items[indx].guard_array
        poly = []
        positions = [[0, 0], [proportions[0], 0], proportions,
                     [0, proportions[1]]]
        for position in positions:
            d = {0: position, 90: np.array([-position[1], position[0]]), 180: -np.array(position),
                 270: np.array([position[1], -position[0]])}
            if not math.isnan(orientation):
              poly.append(np.array(d[orientation] + np.array(self.positions[indx])))
        return np.matrix(poly)


    def area_list(self, indx):
        """ returns the furniture lists of area #index """
        return self.items[indx].furnitures.items


    def __str__(self):
        return 'room areas are: %s' % self.items


class FlatRoom:

    def __init__(self, furnitures, positions, orientations, poly):
        self.furnitures = furnitures
        self.positions = positions
        self.orientations = orientations
        self.poly = poly

    @classmethod
    def init_from_room_class(cls, room):
        poly = room.poly
        orientations = []
        positions = []
        furnitures = []

        for area_indx in range(len(room.items)):
            orientations += list(room.area_orientation(area_indx))  # converting from np.array to list
            positions += list(room.area_pos(area_indx))
            furnitures += list(room.area_list(area_indx))
        # print(cls)
        return cls(furnitures, positions, orientations, poly)

    def pop_furniture(self, index):
        orientation = self.orientations.pop(index)
        position = self.positions.pop(index)
        furniture = self.furnitures.pop(index)
        return orientation, position, furniture

    def get_one_furniture_poly(self, indx, add_guard=False):
        """
        Returns index Furniture's polynom of
        :param indx:
        :return: array of area polynom
        """

        orientation = self.orientations[indx]
        proportions = self.furnitures[indx].proportions
        if add_guard is True:
            proportions += self.furnitures[indx].guard_array
        poly = []
        positions = [[-proportions[0] / 2, -proportions[1] / 2], [-proportions[0] / 2,  proportions[1] / 2],
                     [ proportions[0] / 2,  proportions[1] / 2], [ proportions[0] / 2, -proportions[1] / 2]]
        for position in positions:
            d = {0: position, 90: np.array([-position[1], position[0]]), 180: -np.array(position),
                 270: np.array([position[1], -position[0]])}
            if not math.isnan(orientation):
              poly.append(np.array(d[orientation] + np.array(self.positions[indx])))
        return np.matrix(poly)

    def furnitures_polygons(self, add_guard = False):
        """
        Returns all the room's Furnitures polygons
        :param indx:
        :return: array of area polynom
        """
        furnitures_poly_list = [self.get_one_furniture_poly(ind, add_guard) for ind in range(len(self.furnitures))]
        furnitures_poly = []

        for furn_poly in furnitures_poly_list:
            if len (furn_poly) >= 4:
                poly1 = []
                for point in furn_poly:
                    # poly1.append(pvg.Point(point[0,0], point[0,1]))
                    poly1.append((point[0,0], point[0,1]))
                furnitures_poly.append(poly1)
        return furnitures_poly

    def move_furniture(self, indx, step):
        self.positions[indx] = self.positions[indx] + np.array(step)

    def plot(self, guard=False):
        room_fig = plt.figure()
        for furn_poly in self.furnitures_polygons(guard):
            plt.plot(np.vstack([furn_poly, furn_poly[0]])[:, 0], np.vstack([furn_poly, furn_poly[0]])[:, 1])
        plt.plot(np.vstack([self.poly, self.poly[0]])[:, 0], np.vstack([self.poly, self.poly[0]])[:, 1])
        plt.draw_if_interactive()
        plt.show()

if __name__ == "__main__":
    plt.ioff()
    import room_designer.room_designer
    poly = np.array(
        [[100.0, 150.0], [200.0, 150.0], [200.0, 100.0], [400.0, 100.0], [400.0, 200.0], [750.0, 200.0], [750.0, 150.0],
         [850.0, 150.0], [850.0, 600.0], [700.0, 600.0], [700.0, 700.0], [800.0, 700.0], [800.0, 800.0], [250.0, 800.0],
         [250.0, 750.0],
         [150.0, 750.0], [150.0, 800.0], [100.0, 800.0], [100.0, 600.0], [150.0, 600.0], [150.0, 250.0], [100.0, 250.0],
         [100.0, 150.0]])

    room = room_designer.room_designer.arrange_living_room(poly)
    room.plot()
    room_flat = FlatRoom.init_from_room_class(room)
    print([furntiture.name for furntiture in room_flat.furnitures])
    # furnitures_poly = [room_flat.area_poly(room_flat, ind) for ind in range(len(room_flat.list))]
    # pos = [room_flat.area_pos(room_flat, ind) for ind in range(len(room_flat.list))]

    polys = room_flat.furnitures_polygons(False)

    polys.append(room_flat.poly)
    print(polys)
    pS = [[151.0, 280.0]]
    pT = [[800.0, 280.0]]

    #plot polys

    room_flat.plot()

    pathRatioSum2 = layoutCostFunctions.calcLayoutCirculation(polys, pS, pT)
    print('path ratio: {:.3}'.format(pathRatioSum2))
    polys.pop(6)
    polys[6] = [tuple(np.array(point) + np.array((0, 90))) for point in polys[6]]
    clearance = layoutCostFunctions.calcLayoutClearance(polys[0:len(polys) - 1], polys[-1], pS)
    print('clearance: {:.3}'.format(clearance))

    dR = 50
    positions = [pos for pos in room_flat.positions if pos is not np.nan]
    positions.pop(6)
    group_relation = layoutCostFunctions.calcGroupRelation(positions, [1] * len(polys), dR)
    print('group relation: {:.3}'.format(group_relation))
