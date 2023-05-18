import gym
import matplotlib.pyplot as plt
import numpy as np
from shapely import geometry

class SocialNavWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.pedestrians = None
        self.robot = None
        self.obstacles = None
        self.time = 0.0
        self.goal = None
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.pedestrians = self.parse_predestrian(observation, info)
        self.robot = self.parse_robot(observation, info)
        self.obstacles = self.parse_obstacles(observation, info)
        self.goal = self.parse_goal(observation, info)
        self.time += self.get_timestep(info) 
    
    def reset(self):
        return self.env.reset()
    
    def parse_predestrian(self, observation, info):
        pedestrian_data = info["pedestrian_data"]
        pedestrians = []
        for data in pedestrian_data:
            position = data["position"]
            goal = data["goal"]
            pedestrians.append(Pedestrian(position, goal))
        return pedestrians
    
    def parse_robot(self, observation, info):
        robot_data = info["robot_data"]
        position = robot_data["position"]
        return Robot(position)
    
    def parse_obstacles(self, observation, info):
        obstacle_data = info["obstacle_data"]
        obstacles = []
        for data in obstacle_data:
            points = data["points"]
            obstacles.append(Obstacle(points))
        return obstacles
    
    def parse_goal(self, observation, info):
        return info["goal"]
    
    def get_timestep(self, info):
        return info["timestep"]
    
    def get_metrics(self):
        distance = np.linalg.norm(self.robot.position - self.goal)
        return {"distance_to_goal": distance}
    
    def render(self):
        plt.clf()
        for pedestrian in self.pedestrians:
            plt.scatter(pedestrian.position[0], pedestrian.position[1], color="b")
        plt.scatter(self.robot.position[0], self.robot.position[1], color="r")
        for obstacle in self.obstacles:
           x, y = obstacle.poly.exterior.xy
            plt.plot(x, y, color="k")
        plt.show(block=False)
        plt.pause(0.1)

class Pedestrian:
    def __init__(self, position, goal):
        self.position = np.array(position)
        self.goal = np.array(goal)

class Robot:
    def __init__(self, position):
        self.position = np.array(position)

class Obstacle:
    def __init__(self, points):
        self.poly = geometry.Polygon([[p[0], p[1]] for p in points])
