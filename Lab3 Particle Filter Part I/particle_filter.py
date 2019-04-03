#################################
# Ting Qiu
# George Jeno
#################################

from grid import *
from particle import Particle
from utils import *
from setting import *
import setting
import numpy as np


def motion_update(particles,odom):
    """ Particle filter motion update

        Arguments:
        particles -- input list of particle represents belief p(x_{t-1} | u_{t-1})
                before motion update
        odom -- odometry to move (dx,dy,dh) in *robot local frame*

        Returns: the list of particles represents belief \tilde{p}(x_{t} | u_{t})
                after motion update
    """
    motion_particles = []
    for p in particles:
        dx,dy,dh = add_odometry_noise(odom,ODOM_HEAD_SIGMA,ODOM_TRANS_SIGMA)
        dx,dy = rotate_point(dx,dy,p.h)
        p.x = p.x + dx
        p.y = p.y + dy
        p.h = p.h + dh
        motion_particles.append(p)
    return motion_particles

# ------------------------------------------------------------------------
def measurement_update(particles,measured_marker_list,grid):
    """ Particle filter measurement update

        Arguments:
        particles -- input list of particle represents belief \tilde{p}(x_{t} | u_{t})
                before meansurement update (but after motion update)

        measured_marker_list -- robot detected marker list,each marker has format:
                measured_marker_list[i] = (rx,ry,rh)
                rx -- marker's relative X coordinate in robot's frame
                ry -- marker's relative Y coordinate in robot's frame
                rh -- marker's relative heading in robot's frame,in degree

                * Note that the robot can only see markers which is in its camera field of view,
                which is defined by ROBOT_CAMERA_FOV_DEG in setting.py
				* Note that the robot can see mutliple markers at once,and may not see any one

        grid -- grid world map,which contains the marker information,
                see grid.py and CozGrid for definition
                Can be used to evaluate particles

        Returns: the list of particles represents belief p(x_{t} | u_{t})
                after measurement update
    """
    measured_particles = []
    count = 0
    sum = 1
    weight = []

    if len(measured_marker_list) == 0:
        for p in particles:
            weight.append((p,1/len(particles)))
    else:
        for p in particles:
            if not grid.is_in(p.x,p.y):
                weight.append((p,0))
            elif not grid.is_free(p.x,p.y):
                weight.append((p,0))
            else:
                pairs = []
                read_marker_to_p = p.read_markers(grid)
                difference = math.fabs(len(measured_marker_list) - len(read_marker_to_p))
                for mm in measured_marker_list:
                    if len(read_marker_to_p) != 0:
                        mmx,mmy,mmh = add_marker_measurement_noise(mm,MARKER_TRANS_SIGMA,MARKER_ROT_SIGMA)
                        m = read_marker_to_p[0]
                        dist = grid_distance(mmx,mmy,m[0],m[1])
                        for pm in read_marker_to_p:
                            pm_dist = grid_distance(mmx,mmy,pm[0],pm[1])
                            if dist > pm_dist:
                                dist = pm_dist
                                m = pm
                        pairs.append((m,mm))
                        read_marker_to_p.remove(m)
                    else:
                        break

                prob = 1
                longest_distance = 0
                for p_marker,mm in pairs:
                    prob *= np.exp(-(grid_distance(p_marker[0],p_marker[1],mm[0],mm[1])** 2/(2 *MARKER_TRANS_SIGMA *2)+diff_heading_deg(p_marker[2],mm[2])** 2 /(2 * MARKER_ROT_SIGMA ** 2)))
                    if grid_distance(p_marker[0],p_marker[1],mm[0],mm[1]) > longest_distance:
                        longest_distance = grid_distance(p_marker[0],p_marker[1],mm[0],mm[1])
                for i in range(int(difference)):
                    prob = prob *np.exp(-((longest_distance**2)/(2*(MARKER_TRANS_SIGMA**2))+(45**2)/(2*(MARKER_ROT_SIGMA**2))))
                weight.append((p,prob))
        weight = sorted(weight,key=lambda weight : weight[1])
        sum = 0
        remove = int(PARTICLE_COUNT/120)
        weight = weight[remove:]
        for a,b in weight:
            if b == 0:
                count =count+ 1
            else:
                sum = sum + b
        weight = weight[count:]
        count = count +remove
    particles_List = []
    weight1 = []
    for a,b in weight:
        particles_List.append(Particle(a.x,a.y,a.h))
        new_weight = b / sum
        weight1.append(new_weight)

    random = Particle.create_random(count,grid)
    for random_p in random:
        measured_particles.append(random_p)
    particle_List1 = []
    if particles_List != []:
        particle_List1 = np.random.choice(particles_List,size =len(particles_List),replace = True,p=weight1)
    for particle in particle_List1:
        px,py,ph= add_odometry_noise([particle.x,particle.y,particle.h],ODOM_HEAD_SIGMA,ODOM_TRANS_SIGMA)
        measured_particles.append(Particle(px,py,ph))

    return measured_particles
