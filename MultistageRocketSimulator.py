import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

G = 6.67E-11
MEarth = 5.97E24
REarth = 6.38E6
C_d = 0.2
H = 8000
rho_0 = 1.2225
dt = 0.5 #integrate at steps of 0.5 seconds, can decrease dt for greater accuracy

class Stage:
    def __init__(self, stage_info): #stage_info should be 5-item array of wet mass, dry mass, burn time, ejection velocity, and diameter
        self.wet_mass = stage_info[0]
        self.dry_mass = stage_info[1]
        self.burn_time = stage_info[2]
        self.u = stage_info[3]
        self.diameter = stage_info[4]
        
class Rocket:
    def __init__(self, rocket_info): #rocket_info should be array of Stages
        self.stages = rocket_info
        self.num_stages = len(rocket_info)
        total_mass = 0
        for i in range(self.num_stages):
            total_mass += self.stages[i].wet_mass
            total_mass += self.stages[i].dry_mass
        self.total_mass = total_mass
        
        total_time = 0
        for i in range(self.num_stages):
            total_time += self.stages[i].burn_time
        num_data_points = round(total_time / dt)
        self.total_time = total_time
        
    def M(self, t):
        index = 0
        total_burn_time = 0
        while (index < self.num_stages):
            total_burn_time += self.stages[index].burn_time
            if (t > total_burn_time):
                index += 1
            else:
                break
        stage_num = index
        index = 0
        current_mass = self.total_mass
        current_time = 0
        while (index < stage_num):
            current_mass -= self.stages[index].wet_mass + self.stages[index].dry_mass
            current_time += self.stages[index].burn_time
            index += 1
        time_in_stage = t - current_time
        dMdt = -self.stages[stage_num].wet_mass/self.stages[stage_num].burn_time
        current_mass -= time_in_stage * -dMdt
        return current_mass 
    
    def D(self, t):
        index = 0
        total_burn_time = 0
        while (index < self.num_stages):
            total_burn_time += self.stages[index].burn_time
            if (t > total_burn_time):
                index += 1
            else:
                break
        stage_num = index
        max_diam = self.stages[stage_num].diameter
        while (stage_num < self.num_stages):
            if (self.stages[stage_num].diameter > max_diam):
                max_diam = self.stages[stage_num].diameter
            stage_num += 1
        return max_diam
    
    def system(self, t, status, neglect_other_forces): #helper function, takes in [v,h] and returns [dvdt, dhdt]
        v = status[0]
        h = status[1]
        index = 0
        total_burn_time = 0
        while (index < self.num_stages):
            total_burn_time += self.stages[index].burn_time
            if (t > total_burn_time):
                index += 1
            else:
                break
        stage_num = index
        dMdt = -self.stages[stage_num].wet_mass/self.stages[stage_num].burn_time
        propulsion_term = (self.stages[stage_num].u/self.M(t))*(-dMdt)
        grav_term = (G * MEarth)/((REarth + h)**2)
        rho = rho_0 * math.exp(-h/H)
        A = math.pi * ((self.stages[stage_num].diameter/2)**2)
        drag_term = (1/(2*self.M(t))) * rho * (v**2) * A * C_d
        if (neglect_other_forces is False):
            dvdt = propulsion_term - grav_term - drag_term
        else:
            dvdt = propulsion_term
        dhdt = v
        return [dvdt, dhdt]

    def simulate(self):
        initial = [0,0] #initial height and velocity both zero
        sol_all_forces = solve_ivp(self.system, (0, self.total_time), initial, args = [False], method = 'RK45', dense_output = 'true', max_step = dt) #use 4th-order Runge-Kutta method
        self.v = sol_all_forces.y[0,:]
        self.h = sol_all_forces.y[1,:]
        self.t = sol_all_forces.t
        sol_only_propulsion = solve_ivp(self.system, (0, self.total_time), initial, args = [True], method = 'RK45', dense_output = 'true', max_step = dt)
        self.v_without_other_forces = sol_only_propulsion.y[0,:]
        self.h_without_other_forces = sol_only_propulsion.y[1,:]
        
    def return_forces(self, t, v, h): #used for plotting the varying acceleration due to propulsion, drag, and gravity, over time
        index = 0
        total_burn_time = 0
        while (index < self.num_stages):
            total_burn_time += self.stages[index].burn_time
            if (t > total_burn_time):
                index += 1
            else:
                break
        stage_num = index
        dMdt = -self.stages[stage_num].wet_mass/self.stages[stage_num].burn_time
        propulsion_term = (self.stages[stage_num].u/self.M(t))*(-dMdt)
        grav_term = (G * MEarth)/((REarth + h)**2)
        rho = rho_0 * math.exp(-h/H)
        A = math.pi * ((self.stages[stage_num].diameter/2)**2)
        drag_term = (1/(2*self.M(t))) * rho * (v**2) * A * C_d
        return [propulsion_term, grav_term, drag_term]
    
    def plot(self):
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1)
        fig.set_figheight(30)
        fig.set_figwidth(15)
        
        ax1.plot(self.t, self.v)
        ax1.plot(self.t, self.v_without_other_forces)
        ax1.legend(['With Gravity and Drag','Without Gravity and Drag'])
        ax1.set_ylabel('Velocity (m/s)')
        ax1.set_xlabel('Time (s)')
        
        ax2.plot(self.t, self.h)
        ax2.plot(self.t, self.h_without_other_forces)
        ax2.legend(['With Gravity and Drag','Without Gravity and Drag'])
        ax2.set_ylabel('Height (m)')
        ax2.set_xlabel('Time (s)')
        
        size = self.t.shape[0]
        self.propulsion_acc = np.zeros(size)
        self.grav_acc = np.zeros(size)
        self.drag_acc = np.zeros(size)
        
        for i in range(0, size):
            acc_due_to_forces = self.return_forces(self.t[i], self.v[i], self.h[i])
            self.propulsion_acc[i] = acc_due_to_forces[0]
            self.grav_acc[i] = acc_due_to_forces[1]
            self.drag_acc[i] = acc_due_to_forces[2]
        
        ax3.plot(self.t, self.propulsion_acc)
        ax3.set_ylabel('Acceleration due to Propulsion (m/s^2)')
        ax3.set_xlabel('Time (s)')
        ax4.plot(self.t, self.grav_acc)
        ax4.set_ylabel('Minus Acceleration due to Gravity (m/s2)')
        ax4.set_xlabel('Time (s)')
        ax5.plot(self.t, self.drag_acc)
        ax5.set_ylabel('Minus Acceleration due to Drag')
        ax5.set_xlabel('Time (s)')
        
        
        
Stage_1 = Stage(np.array([2169000, 131000, 168, 2556.16, 10]))
Stage_2 = Stage(np.array([444000, 36000, 384, 4446.27, 10]))
Stage_3 = Stage(np.array([109000, 10000, 494, 8141.43, 6.604]))
Saturn_V_Stages = np.array([Stage_1, Stage_2, Stage_3])
Saturn_V_Rocket = Rocket(Saturn_V_Stages)
Saturn_V_Rocket.simulate()
Saturn_V_Rocket.plot()
