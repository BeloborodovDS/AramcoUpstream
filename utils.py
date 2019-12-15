import numpy as np
from matplotlib import pyplot as plt
import sklearn
from sklearn.neural_network import MLPRegressor
import itertools
from scipy.optimize import linear_sum_assignment
import gc
import time



def assign_targets(points_xy, valid_mask, drones_xy, base_xy, drones_death_dist, net, death_thresh):
    if not np.any(valid_mask):
        return np.array([]), np.array([]), np.array([])
    
    points_xy = points_xy[valid_mask]
    
    num_points = points_xy.shape[0]
    num_drones = drones_xy.shape[0]
    # num_drones x num_points
    dist_drn_pnt = np.sqrt(np.sum((points_xy[None,:,:] - drones_xy[:,None,:])**2, axis=2))
    dist_drn_bs = np.sqrt(np.sum((drones_xy - base_xy)**2, axis=1))
    to_base = (dist_drn_bs + death_thresh >= drones_death_dist)
    dist_drn_bs = dist_drn_bs[:,None] + np.zeros_like(dist_drn_pnt)
    dist_pnt_bs = np.sqrt(np.sum((points_xy - base_xy)**2, axis=1))[None,:] + np.zeros_like(dist_drn_pnt)
    death = drones_death_dist[:,None] + np.zeros_like(dist_drn_pnt)
    dist_drn_drn = np.max(np.sqrt(np.sum((drones_xy[None,:,:]-drones_xy[:,None,:])**2, axis=2)), axis=1)
    dist_drn_drn = dist_drn_drn[:,None] + np.zeros_like(dist_drn_pnt)
    # num_drones*num_points x 4
    feats = np.hstack([dist_drn_pnt.reshape(-1,1), dist_drn_bs.reshape(-1,1), 
                       dist_pnt_bs.reshape(-1,1), death.reshape(-1,1), dist_drn_drn.reshape(-1,1)])
    
    feats /= 2000.
    # num_drones*num_points x 1
    if net is None:
        scores = feats[:,0].reshape((num_drones, num_points))
    else:
        scores = net.forward(feats).reshape((num_drones, num_points))
    
    scores[to_base,:] = np.max(scores) + 1
    drones, points = linear_sum_assignment(scores)
    
    to_base = to_base[drones]
    points[to_base] = -1
    
    tmp = np.arange(valid_mask.size)[valid_mask]
    points = np.array([-1 if p==-1 else tmp[p] for p in points])
    
    fullpoints = - 2 * np.ones(drones_xy.shape[0], dtype=int)
    fullpoints[drones] = points
    
    return drones, points, fullpoints


class Drone():
    def __init__(self, xy, base_xy, speed=13.88888, 
                 fly_discharge=0.00055555, msr_discharge=2.77778e-05, charge_time=600):
        self.speed = speed
        self.xy = xy.copy()
        self.base_xy = base_xy.copy()
        self.fly_discharge = fly_discharge
        self.msr_discharge = msr_discharge
        self.charge = 1.
        self.dt = 6.
        self.eps = 800 #85*4
        self.timeout = 0
        self.charge_steps = charge_time//self.dt
        self.landed = False
        self.update_death()
        self.c = 'y'
        
    def update_death(self):
        self.base_dist = np.sqrt(np.sum((self.xy-self.base_xy)**2))
        self.death_dist = self.charge/self.fly_discharge*self.speed
        self.returning = (self.death_dist <= self.base_dist + self.eps)
        
    def move_to(self, target, target_index):
        if self.timeout > 0:
            #self.timeout -= 1
            self.c = 'r'
            if self.timeout == 1:
                self.target_index = -3
                self.c = 'y'
            return
        
        self.charge -= self.fly_discharge*self.dt
        self.charge = max(self.charge, 0.)
        self.update_death()
        
        if self.returning:
            target = self.base_xy.copy()
            target_index = -1
            self.landed = False
        self.target = target
        self.target_index = target_index
        dist = np.sqrt(np.sum((self.xy-target)**2))

        if dist <= self.dt*self.speed:
            self.xy = target.copy()
            if target_index != -3:
                self.landed = True
            if target_index==-1:
                self.charge = 1.
                self.returning = False
                self.timeout = int(self.charge_steps)
                self.landed = False
            return
        if (self.landed or target_index==-2): #if landed or no tasks available
            return
        direct = target-self.xy
        direct = direct/np.sqrt(np.sum(direct**2))
        self.xy += direct*self.speed*self.dt
        
    def measure(self, i, N):
        if self.landed:
            self.charge -= (self.fly_discharge + self.msr_discharge)*self.dt
            self.charge = max(self.charge, 0)
            self.update_death()
            self.c = 'y' #['r','g','b','m'][i]
            if i==N-1:
                self.landed = False
                self.c = 'y'
            return self.target_index
        else:
            self.move_to(self.target, self.target_index)
            return -1
        
        
        

class Environment():
    def __init__(self, area_x=(4500,6500), area_y=(300,900), grid_x=41, grid_y=13,
                 base_xy=(5000.0,1000), num_drones=5, drones_xy=None, drone_speed=13.88888,
                 car_xy=(0,50), car_speed=1.38888, car_steps=[10,3,1], 
                 fly_discharge=0.00055555, msr_discharge=2.77778e-05, charge_time=600):
        self.i = 0
        self.dt = 6.
        self.grid_x = grid_x
        self.grid_x = grid_y
        self.area_x = area_x
        self.area_y = area_y
        x = np.zeros((grid_y,grid_x)) + np.linspace(area_x[0], area_x[1], grid_x)[None,:]
        y = np.zeros((grid_y,grid_x)) + np.linspace(area_y[0], area_y[1], grid_y)[:,None]
        self.points = np.hstack([x.reshape(-1,1), y.reshape(-1,1)])
        self.valid_mask = np.ones(self.points.shape[0], dtype=bool)
        
        self.car_xy = np.array(car_xy)[None]
        self.car_steps = np.cumsum([0] + car_steps)
        self.car_speed = car_speed
        self.new_points = self.get_drone_points()
        
        self.base_xy = np.array(base_xy)[None]
        if drones_xy is None:
            drones_xy = 10*np.random.normal(0,1,(num_drones,2)) + self.base_xy
        self.drones = [Drone(d[None], self.base_xy) for d in drones_xy]        
        
        self.sending = False
        
        self.meandist = []
        
    def get_data(self):
        drones_xy = np.vstack([d.xy for d in self.drones])
        death = np.array([d.death_dist for d in self.drones])
        
        return [self.new_points, self.valid_mask, drones_xy, self.base_xy, death]
        
    def get_drone_points(self):
        j = self.i % self.car_steps[-1]
        xy = self.car_xy.copy()
        xy[0,0] += max(self.car_steps[1]-j, 0)*self.car_speed*self.dt
        return 2*(self.points - xy) + xy
    
    def recharge_drones(self):
        recharging = [d for d in self.drones if d.timeout > 0]
        if len(recharging)>=2:
            recharging[1].timeout -= 1
        if len(recharging)>=1:
            recharging[0].timeout -= 1
    
    def step(self, assignment):
        self.recharge_drones()
        lo = self.car_steps[:-1]
        hi = self.car_steps[1:]
        j = self.i % self.car_steps[-1]
        if lo[0] <= j < hi[0]:
            self.car_xy[0, 0] += self.car_speed*self.dt
            if (self.car_xy[0, 0] >= 10000) or (self.car_xy[0, 0] < 0):
                self.car_speed *= -1
            self.sending = False
        elif lo[1] <= j < hi[1]:
            self.sending = True
        else:
            self.sending = True
            
        self.new_points = self.get_drone_points()
        
        mdst = np.mean(np.concatenate([[np.sqrt(np.sum((self.drones[i].xy-self.drones[j].xy)**2)) 
          for j in range(i)] 
         for i in range(len(self.drones))]))
        self.meandist.append(mdst)
            
        if self.sending:
            measured = np.array([d.measure((j-lo[1])%(hi[2]-lo[1]), hi[2]-lo[1]) for d in self.drones])
            measured = measured[measured >= 0]
            if measured.size>0:
                self.valid_mask[measured] = False
        else:
            for d,a in zip(self.drones, assignment):
                d.move_to(self.new_points[a][None], a)
            
        self.i += 1

        # print([d.charge for d in self.drones])
        # print([d.returning for d in self.drones])
        # print([d.landed for d in self.drones])
        # print([d.timeout for d in self.drones])
        # print([d.target for d in self.drones])
        # print([d.xy for d in self.drones])
        # print ' '
        
    def visualize(self):
        #plt.figure(figsize=(7,7))
        plt.scatter(self.points[:,0], self.points[:,1], marker='.')
        c = 'g' if self.sending else 'r'
        pnt = self.new_points
        plt.scatter(pnt[:,0], pnt[:,1], color=c, marker='x', alpha=0.5)
        plt.scatter(pnt[~self.valid_mask,0], pnt[~self.valid_mask,1], color='m', alpha=0.5)
        plt.scatter([self.car_xy[0,0]], [self.car_xy[0,1]], color='k', marker='D')
        plt.scatter([self.base_xy[0,0]], [self.base_xy[0,1]], color='k', marker='*')
        for d in self.drones:
            c = ('k' if d.landed else 'y')
            plt.scatter([d.xy[0,0]], [d.xy[0,1]], color=d.c, marker='>', alpha=(d.charge+1.0)/2)
        plt.xlim(0,10000)
        plt.ylim(0,2000)
        plt.show(block=False)
        plt.pause(0.01)
        plt.clf()
        #plt.close()
        gc.collect()
        
        
        
        
        
