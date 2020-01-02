from rooms import FlatRoom
import pickle
import layoutCostFunctions


file_obj = open('room_example.pk1', 'rb')
saved_room = pickle.load(file_obj)

updated_room = FlatRoom(saved_room.furnitures, saved_room.positions, saved_room.orientations, saved_room.poly)
updated_room.plot()


# run basic cost function on room
polys = updated_room.furnitures_polygons(False)

polys.append(updated_room.poly)
# print(polys)
pS = [[151.0, 280.0]]
pT = [[800.0, 280.0]]

pathRatioSum2 = layoutCostFunctions.calcLayoutCirculation(polys, pS, pT)
print('path ratio: {:.3}'.format(pathRatioSum2))

pathRatioSum2 = layoutCostFunctions.calcLayoutCirculation(polys, pS, pT)
print('path ratio: {:.3}'.format(pathRatioSum2))
clearance = layoutCostFunctions.calcLayoutClearance(polys[0:len(polys) - 1], polys[-1], pS)
print('clearance: {:.3}'.format(clearance))

dR = 50
# positions = [pos for pos in updated_room.positions if pos is not np.nan]
# positions.pop(6)
group_relation = layoutCostFunctions.calcGroupRelation(updated_room.positions, [1] * len(polys), dR)
print('group relation: {:.3}'.format(group_relation))