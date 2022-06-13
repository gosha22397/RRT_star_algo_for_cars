from __future__ import division
from logging import raiseExceptions
from shapely.geometry import Point, LineString
from IPython.display import clear_output
from enum import Enum
import random
import math
import sys
import numpy as np

class TurnType(Enum):
    LSL = 1
    LSR = 2
    RSL = 3
    RSR = 4
    RLR = 5
    LRL = 6

class Param:
    def __init__(self, p_init, seg_final, turn_radius):
        self.p_init = p_init
        self.seg_final = seg_final
        self.turn_radius = turn_radius
        self.type = 0

def wrapTo360(angle):
    posIn = angle>0
    angle = angle % 360
    if angle == 0 and posIn:
        angle = 360
    return angle

def wrapTo180(angle):
    q = (angle < -180) or (180 < angle)
    if(q):
        angle = wrapTo360(angle + 180) - 180
    return angle

def headingToStandard(hdg):
    thet = wrapTo360(90 - wrapTo180(hdg))
    return thet

def calcDubinsPath(wpt1, wpt2, turn_rad):
    param = Param(wpt1, 0, 0)
    tz        = [0, 0, 0, 0, 0, 0]
    pz        = [0, 0, 0, 0, 0, 0]
    qz        = [0, 0, 0, 0, 0, 0]
    param.seg_final = [0, 0, 0]

    psi1 = headingToStandard(wpt1[2])*math.pi/180
    psi2 = headingToStandard(wpt2[2])*math.pi/180

    param.turn_radius = turn_rad
    dx = wpt2[0] - wpt1[0]
    dy = wpt2[1] - wpt1[1]
    D = math.sqrt(dx*dx + dy*dy)
    d = D/param.turn_radius

    theta = math.atan2(dy,dx) % (2*math.pi)
    alpha = (psi1 - theta) % (2*math.pi)
    beta  = (psi2 - theta) % (2*math.pi)
    best_word = -1
    best_cost = -1

    tz[0], pz[0], qz[0] = dubinsLSL(alpha,beta,d)
    tz[1], pz[1], qz[1] = dubinsLSR(alpha,beta,d)
    tz[2], pz[2], qz[2] = dubinsRSL(alpha,beta,d)
    tz[3], pz[3], qz[3] = dubinsRSR(alpha,beta,d)
    tz[4], pz[4], qz[4] = dubinsRLR(alpha,beta,d)
    tz[5], pz[5], qz[5] = dubinsLRL(alpha,beta,d)

    for x in range(6):
        if(tz[x]!=-1):
            cost = tz[x] + pz[x] + qz[x]
            if(cost<best_cost or best_cost==-1):
                best_word = x+1
                best_cost = cost
                param.seg_final = [tz[x],pz[x],qz[x]]

    param.type = TurnType(best_word)
    return param

def dubinsLSL(alpha, beta, d):
    tmp0      = d + math.sin(alpha) - math.sin(beta)
    tmp1      = math.atan2((math.cos(beta)-math.cos(alpha)),tmp0)
    p_squared = 2 + d*d - (2*math.cos(alpha-beta)) + (2*d*(math.sin(alpha)-math.sin(beta)))
    if p_squared<0:
        p=-1
        q=-1
        t=-1
    else:
        t         = (tmp1-alpha) % (2*math.pi)
        p         = math.sqrt(p_squared)
        q         = (beta - tmp1) % (2*math.pi)
    return t, p, q

def dubinsRSR(alpha, beta, d):
    tmp0      = d - math.sin(alpha) + math.sin(beta)
    tmp1      = math.atan2((math.cos(alpha)-math.cos(beta)),tmp0)
    p_squared = 2 + d*d - (2*math.cos(alpha-beta)) + 2*d*(math.sin(beta)-math.sin(alpha))
    if p_squared<0:
        p=-1
        q=-1
        t=-1
    else:
        t         = (alpha - tmp1 ) % (2*math.pi)
        p         = math.sqrt(p_squared)
        q         = (-1*beta + tmp1) % (2*math.pi)
    return t, p, q

def dubinsRSL(alpha,beta,d):
    tmp0      = d - math.sin(alpha) - math.sin(beta)
    p_squared = -2 + d*d + 2*math.cos(alpha-beta) - 2*d*(math.sin(alpha) + math.sin(beta))
    if p_squared<0:
        p=-1
        q=-1
        t=-1
    else:
        p         = math.sqrt(p_squared)
        tmp2      = math.atan2((math.cos(alpha)+math.cos(beta)),tmp0) - math.atan2(2,p)
        t         = (alpha - tmp2) % (2*math.pi)
        q         = (beta - tmp2) % (2*math.pi)
    return t, p, q

def dubinsLSR(alpha, beta, d):
    tmp0      = d + math.sin(alpha) + math.sin(beta)
    p_squared = -2 + d*d + 2*math.cos(alpha-beta) + 2*d*(math.sin(alpha) + math.sin(beta))
    if p_squared<0:
        p=-1
        q=-1
        t=-1
    else:
        p         = math.sqrt(p_squared)
        tmp2      = math.atan2((-1*math.cos(alpha)-math.cos(beta)),tmp0) - math.atan2(-2,p)
        t         = (tmp2 - alpha) % (2*math.pi)
        q         = (tmp2 - beta) % (2*math.pi)
    return t, p, q

def dubinsRLR(alpha, beta, d):
    tmp_rlr = (6 - d*d + 2*math.cos(alpha-beta) + 2*d*(math.sin(alpha)-math.sin(beta)))/8
    if(abs(tmp_rlr)>1):
        p=-1
        q=-1
        t=-1
    else:
        p = (2*math.pi - math.acos(tmp_rlr)) % (2*math.pi)
        t = (alpha - math.atan2((math.cos(alpha)-math.cos(beta)), d-math.sin(alpha)+math.sin(beta)) + p/2 % (2*math.pi)) % (2*math.pi)
        q = (alpha - beta - t + (p % (2*math.pi))) % (2*math.pi)
    return t, p, q

def dubinsLRL(alpha, beta, d):
    tmp_lrl = (6 - d*d + 2*math.cos(alpha-beta) + 2*d*(-1*math.sin(alpha)+math.sin(beta)))/8
    if(abs(tmp_lrl)>1):
        p=-1
        q=-1
        t=-1
    else:
        p = (2*math.pi - math.acos(tmp_lrl)) % (2*math.pi)
        t = (-1*alpha - math.atan2((math.cos(alpha)-math.cos(beta)), d+math.sin(alpha)-math.sin(beta)) + p/2) % (2*math.pi)
        q = ((beta % (2*math.pi))-alpha-t+(p % (2*math.pi))) % (2*math.pi)
    return t, p, q

def dubins_length(param):
    return (param.seg_final[0]+param.seg_final[1]+param.seg_final[2])*param.turn_radius

def dubins_path(param, length):
    tprime = length/param.turn_radius
    p_init = np.array([0,0,headingToStandard(param.p_init[2])*math.pi/180])

    L_SEG = 1
    S_SEG = 2
    R_SEG = 3
    DIRDATA = np.array([[L_SEG,S_SEG,L_SEG],[L_SEG,S_SEG,R_SEG],[R_SEG,S_SEG,L_SEG],[R_SEG,S_SEG,R_SEG],[R_SEG,L_SEG,R_SEG],[L_SEG,R_SEG,L_SEG]])

    types = DIRDATA[param.type.value-1][:]
    param1 = param.seg_final[0]
    param2 = param.seg_final[1]
    mid_pt1 = dubins_segment(param1,p_init,types[0])
    mid_pt2 = dubins_segment(param2,mid_pt1,types[1])

    if(tprime<param1):
        end_pt = dubins_segment(tprime,p_init,types[0])
    elif(tprime<(param1+param2)):
        end_pt = dubins_segment(tprime-param1,mid_pt1,types[1])
    else:
        end_pt = dubins_segment(tprime-param1-param2, mid_pt2, types[2])
    
    end_pt[0] = end_pt[0] * param.turn_radius + param.p_init[0]
    end_pt[1] = end_pt[1] * param.turn_radius + param.p_init[1]
    end_pt[2] = end_pt[2] % (2*math.pi)
    return end_pt

def dubins_segment(seg_param, seg_init, seg_type):
    L_SEG = 1
    S_SEG = 2
    R_SEG = 3
    seg_end = np.array([0.0,0.0,0.0])
    if( seg_type == L_SEG ):
        seg_end[0] = seg_init[0] + math.sin(seg_init[2]+seg_param) - math.sin(seg_init[2])
        seg_end[1] = seg_init[1] - math.cos(seg_init[2]+seg_param) + math.cos(seg_init[2])
        seg_end[2] = seg_init[2] + seg_param
    elif( seg_type == R_SEG ):
        seg_end[0] = seg_init[0] - math.sin(seg_init[2]-seg_param) + math.sin(seg_init[2])
        seg_end[1] = seg_init[1] + math.cos(seg_init[2]-seg_param) - math.cos(seg_init[2])
        seg_end[2] = seg_init[2] - seg_param
    elif( seg_type == S_SEG ):
        seg_end[0] = seg_init[0] + math.cos(seg_init[2]) * seg_param
        seg_end[1] = seg_init[1] + math.sin(seg_init[2]) * seg_param
        seg_end[2] = seg_init[2]
    return seg_end

class PathPlanner():
    def initialise(self, environment, bounds, start_pose, goal_region, object_radius, step_distance, turning_radius, final_prob, num_iterations, curve_type, resolution):
        self.env = environment
        self.obstacles = environment.obstacles
        self.bounds = bounds
        self.minx, self.miny, self.maxx, self.maxy = bounds
        self.start_pose = start_pose
        self.goal_region = goal_region
        self.obj_radius = object_radius
        self.N = num_iterations
        self.resolution = resolution
        self.step_distance = step_distance
        self.turning_radius = turning_radius
        self.final_prob = final_prob
        self.V = set()
        self.E = set()
        self.child_to_parent_dict = dict()
        self.goal_pose = (goal_region.centroid.coords[0])
        self.curve_type = curve_type

    def path(self, environment, bounds, start_pose, goal_region, object_radius, step_distance, turning_radius, final_prob, num_iterations, curve_type, resolution, RRT_Flavour):
        self.env = environment
        self.initialise(environment, bounds, start_pose, goal_region, object_radius, step_distance, turning_radius, final_prob, num_iterations, curve_type, resolution)
        x0, y0 = start_pose[0], start_pose[1]
        x1, y1 = goal_region.centroid.coords[0]
        start = (x0, y0)
        goal = (x1, y1)

        if start == goal:
            path = [start, goal]
            self.V.union([start, goal])
            self.E.union([(start, goal)])
        elif self.isEdgeCollisionFree(start, goal):
            path = [start, goal]
            self.V.union([start, goal])
            self.E.union([(start, goal)])
        else:
            if RRT_Flavour == "RRT":
                path, self.V, self.E = self.RRTSearch()
            elif RRT_Flavour == "RRT*":
                path, self.V, self.E = self.RRTStarSearch()
            else:
                return None, None, None
        return path, self.V, self.E
    
    def RRTSearch(self):
        path = []
        path_length = float('inf')
        tree_size = 0
        path_size = 0
        self.V.add(self.start_pose)
        goal_centroid = self.get_centroid(self.goal_region)

        for i in range(self.N):
            if i % 100 == 0:
                print("RRT, iteration", i, "of", self.N)
            if i + 1 == self.N:
                clear_output(wait=False)
            if(random.random() >= (1 - self.final_prob / 100)):
                random_point = goal_centroid
            else:
                random_point = self.get_collision_free_random_point()

            nearest_point = self.find_nearest_point(random_point)

            if self.curve_type == "standard":
                new_point = self.steer(nearest_point, random_point)
            elif self.curve_type == "dubins":
                new_point = self.dubins(nearest_point, random_point)
            else:
                raise Exception("Wrong curve_type")

            if self.isEdgeCollisionFree(nearest_point, new_point):
                self.V.add(new_point)
                self.E.add((nearest_point, new_point))
                self.setParent(nearest_point, new_point)
                if self.isAtGoalRegion(new_point):
                    tmp_path, tmp_tree_size, tmp_path_size, tmp_path_length = self.find_path(self.start_pose, new_point)
                    if tmp_path_length < path_length:
                        path_length = tmp_path_length
                        path = tmp_path
                        tree_size = tmp_tree_size
                        path_size = tmp_path_size
        return path, self.V, self.E

    def RRTStarSearch(self):
        path = []
        path_length = float('inf')
        tree_size = 0
        path_size = 0
        self.V.add(self.start_pose)
        goal_centroid = self.get_centroid(self.goal_region)

        for i in range(self.N):
            if i % 100 == 0:
                print("RRT_star, iteration", i, "of", self.N)
            if i + 1 == self.N:
                clear_output(wait=False)
            if(random.random() >= (1 - self.final_prob / 100)):
                random_point = goal_centroid
            else:
                random_point = self.get_collision_free_random_point()

            nearest_point = self.find_nearest_point(random_point)

            if self.curve_type == "standard":
                new_point = self.steer(nearest_point, random_point)
            elif self.curve_type == "dubins":
                new_point = self.dubins(nearest_point, random_point)
            else:
                raise Exception("Wrong curve_type")

            if self.isEdgeCollisionFree(nearest_point, new_point) and new_point not in self.V:
                nearest_set = self.find_nearest_set(new_point)
                min_point = self.find_min_point(nearest_set, nearest_point, new_point)
                self.V.add(new_point)
                self.E.add((min_point, new_point))
                self.setParent(min_point, new_point)
                self.rewire(nearest_set, min_point, new_point)
                if self.isAtGoalRegion(new_point):
                    tmp_path, tmp_tree_size, tmp_path_size, tmp_path_length = self.find_path(self.start_pose, new_point)
                    if tmp_path_length < path_length:
                        path_length = tmp_path_length
                        path = tmp_path
                        tree_size = tmp_tree_size
                        path_size = tmp_path_size
        return path, self.V, self.E

    def find_nearest_set(self, new_point):
        points = set()
        ball_radius = self.find_ball_radius()
        for vertex in self.V:
            dist = self.dist(new_point, vertex)
            if dist < ball_radius:
                points.add(vertex)
        return points

    def find_ball_radius(self):
        unit_ball_volume = math.pi
        n = len(self.V)
        dimensions = 2.0
        gamma = (2**dimensions)*(1.0 + 1.0/dimensions) * (self.maxx - self.minx) * (self.maxy - self.miny)
        ball_radius = min(((gamma/unit_ball_volume) * math.log(n) / n)**(1.0/dimensions), self.step_distance)
        return ball_radius

    def find_min_point(self, nearest_set, nearest_point, new_point):
        min_point = nearest_point
        min_cost = self.cost(nearest_point) + self.linecost(nearest_point, new_point)
        for vertex in nearest_set:
            if self.isEdgeCollisionFree(vertex, new_point):
                temp_cost = self.cost(vertex) + self.linecost(vertex, new_point)
                if temp_cost < min_cost:
                    min_point = vertex
                    min_cost = temp_cost
        return min_point

    def rewire(self, nearest_set, min_point, new_point):
        for vertex in nearest_set - set([min_point]):
            if self.isEdgeCollisionFree(vertex, new_point):
                if self.cost(vertex) > self.cost(new_point) + self.linecost(vertex, new_point):
                    parent_point = self.getParent(vertex)
                    self.E.discard((parent_point, vertex))
                    self.E.discard((vertex, parent_point))
                    self.E.add((new_point, vertex))
                    self.setParent(new_point, vertex)

    def cost(self, vertex):
        path, tree_size, path_size, path_length = self.find_path(self.start_pose, vertex)
        return path_length

    def linecost(self, point1, point2):
        return self.dist(point1, point2)

    def getParent(self, vertex):
        return self.child_to_parent_dict[vertex]

    def setParent(self, parent, child):
        self.child_to_parent_dict[child] = parent

    def get_random_point(self):
        x = self.minx + random.random() * (self.maxx - self.minx)
        y = self.miny + random.random() * (self.maxy - self.miny)
        if self.curve_type == "dubins":
            return (x, y, random.uniform(0, 360 - sys.float_info.epsilon))
        return (x, y)

    def get_collision_free_random_point(self):
        while True:
            point = self.get_random_point()
            buffered_point = Point(point).buffer(self.obj_radius, self.resolution)
            if self.isPointCollisionFree(buffered_point):
                return point

    def isPointCollisionFree(self, point):
        for obstacle in self.obstacles:
            if obstacle.contains(point):
                return False
        return True

    def find_nearest_point(self, random_point):
        closest_point = None
        min_dist = float('inf')
        for vertex in self.V:
            dist = self.dist(random_point, vertex)
            if dist < min_dist:
                min_dist = dist
                closest_point = vertex
        return closest_point

    def isOutOfBounds(self, point):
        if((point[0] - self.obj_radius) < self.minx):
            return True
        if((point[1] - self.obj_radius) < self.miny):
            return True
        if((point[0] + self.obj_radius) > self.maxx):
            return True
        if((point[1] + self.obj_radius) > self.maxy):
            return True
        return False

    def isEdgeCollisionFree(self, point1, point2):
        if self.isOutOfBounds(point2):
            return False
        line = LineString([point1[:2], point2[:2]])
        expanded_line = line.buffer(self.obj_radius, self.resolution)
        for obstacle in self.obstacles:
            if expanded_line.intersects(obstacle):
                return False
        return True

    def steer(self, from_point, to_point):
        fromPoint_buffered = Point(from_point).buffer(self.obj_radius, self.resolution)
        toPoint_buffered = Point(to_point).buffer(self.obj_radius, self.resolution)
        if fromPoint_buffered.distance(toPoint_buffered) < self.step_distance:
            return to_point
        else:
            from_x, from_y = from_point[0], from_point[1]
            to_x, to_y = to_point[0], to_point[1]
            theta = math.atan2(to_y - from_y, to_x - from_x)
            new_point = (from_x + self.step_distance * math.cos(theta), from_y + self.step_distance * math.sin(theta))
            return new_point
      
    def dubins(self, from_point, to_point):
        param = calcDubinsPath(from_point, to_point, self.turning_radius)
        new_point = dubins_path(param, self.step_distance)
        return (new_point[0], new_point[1], wrapTo360(90 - (new_point[2] * 360 / (2 * math.pi))))

    def isAtGoalRegion(self, point):
        buffered_point = Point(point[:2]).buffer(self.obj_radius, self.resolution)
        intersection = buffered_point.intersection(self.goal_region)
        inGoal = intersection.area / buffered_point.area
        return inGoal >= 0.5

    def dist(self, point1, point2):
        if self.curve_type == "standard":
            return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
        elif self.curve_type == "dubins":
            param = calcDubinsPath(point1, point2, self.step_distance)
            return dubins_length(param)
        else:
            raise Exception("Wrong curve_type")

    def find_path(self, start_point, end_point):
        path = [end_point]
        tree_size, path_size, path_length = len(self.V), 1, 0
        current_node = end_point
        previous_node = None
        target_node = start_point
        while current_node != target_node:
            parent = self.getParent(current_node)
            path.append(parent)
            previous_node = current_node
            current_node = parent
            path_length += self.dist(current_node, previous_node)
            path_size += 1
        path.reverse()
        return path, tree_size, path_size, path_length

    def get_centroid(self, region):
        centroid = region.centroid.wkt
        filtered_vals = centroid[centroid.find("(")+1:centroid.find(")")]
        filtered_x = filtered_vals[0:filtered_vals.find(" ")]
        filtered_y = filtered_vals[filtered_vals.find(" ") + 1: -1]
        (x,y) = (float(filtered_x), float(filtered_y))
        if self.curve_type == "dubins":
            return (x ,y, 0)
        return (x, y)
