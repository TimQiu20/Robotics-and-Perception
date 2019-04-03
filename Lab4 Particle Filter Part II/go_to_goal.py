# George Jeno, Ting Qiu 

## If you run into an "[NSApplication _setup] unrecognized selector" problem on macOS,
## try uncommenting the following snippet

# try:
#     import matplotlib
#     matplotlib.use('TkAgg')
# except ImportError:
#     pass

from skimage import color
import cozmo
from cozmo.util import Pose, degrees, speed_mmps, distance_inches, distance_mm
import numpy as np
from numpy.linalg import inv
import threading
import time
import sys
import asyncio
from PIL import Image
import math

from markers import detect, annotator

from grid import CozGrid
from gui import GUIWindow
from particle import Particle, Robot
from setting import *
from particle_filter import *
from utils import *

#particle filter functionality
class ParticleFilter:

    def __init__(self, grid):
        self.particles = Particle.create_random(PARTICLE_COUNT, grid)
        self.grid = grid

    def update(self, odom, r_marker_list):

        # ---------- Motion model update ----------
        self.particles = motion_update(self.particles, odom)

        # ---------- Sensor (markers) model update ----------
        self.particles = measurement_update(self.particles, r_marker_list, self.grid)

        # ---------- Show current state ----------
        # Try to find current best estimate for display
        m_x, m_y, m_h, m_confident = compute_mean_pose(self.particles)
        return (m_x, m_y, m_h, m_confident)

# tmp cache
last_pose = cozmo.util.Pose(0,0,0,angle_z=cozmo.util.Angle(degrees=0))
flag_odom_init = False

# goal location for the robot to drive to, (x, y, theta)
goal = (6,10,0)

# map
Map_filename = "map_arena.json"
grid = CozGrid(Map_filename)
gui = GUIWindow(grid, show_camera=True)
pf = ParticleFilter(grid)

def compute_odometry(curr_pose, cvt_inch=True):
    '''
    Compute the odometry given the current pose of the robot (use robot.pose)

    Input:
        - curr_pose: a cozmo.robot.Pose representing the robot's current location
        - cvt_inch: converts the odometry into grid units
    Returns:
        - 3-tuple (dx, dy, dh) representing the odometry
    '''

    global last_pose, flag_odom_init
    last_x, last_y, last_h = last_pose.position.x, last_pose.position.y, \
        last_pose.rotation.angle_z.degrees
    curr_x, curr_y, curr_h = curr_pose.position.x, curr_pose.position.y, \
        curr_pose.rotation.angle_z.degrees

    dx, dy = rotate_point(curr_x-last_x, curr_y-last_y, -last_h)
    if cvt_inch:
        dx, dy = dx / grid.scale, dy / grid.scale

    return (dx, dy, diff_heading_deg(curr_h, last_h))


async def marker_processing(robot, camera_settings, show_diagnostic_image=False):
    '''
    Obtain the visible markers from the current frame from Cozmo's camera.
    Since this is an async function, it must be called using await, for example:

        markers, camera_image = await marker_processing(robot, camera_settings, show_diagnostic_image=False)

    Input:
        - robot: cozmo.robot.Robot object
        - camera_settings: 3x3 matrix representing the camera calibration settings
        - show_diagnostic_image: if True, shows what the marker detector sees after processing
    Returns:
        - a list of detected markers, each being a 3-tuple (rx, ry, rh)
          (as expected by the particle filter's measurement update)
        - a PIL Image of what Cozmo's camera sees with marker annotations
    '''

    global grid

    # Wait for the latest image from Cozmo
    image_event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)

    # Convert the image to grayscale
    image = np.array(image_event.image)
    image = color.rgb2gray(image)

    # Detect the markers
    markers, diag = detect.detect_markers(image, camera_settings, include_diagnostics=True)

    # Measured marker list for the particle filter, scaled by the grid scale
    marker_list = [marker['xyh'] for marker in markers]
    marker_list = [(x/grid.scale, y/grid.scale, h) for x,y,h in marker_list]

    # Annotate the camera image with the markers
    if not show_diagnostic_image:
        annotated_image = image_event.image.resize((image.shape[1] * 2, image.shape[0] * 2))
        annotator.annotate_markers(annotated_image, markers, scale=2)
    else:
        diag_image = color.gray2rgb(diag['filtered_image'])
        diag_image = Image.fromarray(np.uint8(diag_image * 255)).resize((image.shape[1] * 2, image.shape[0] * 2))
        annotator.annotate_markers(diag_image, markers, scale=2)
        annotated_image = diag_image

    return marker_list, annotated_image


async def run(robot: cozmo.robot.Robot):

    global flag_odom_init, last_pose
    global grid, gui, pf

    # start streaming
    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = False
    robot.camera.enable_auto_exposure()
    await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()

    # Obtain the camera intrinsics matrix
    fx, fy = robot.camera.config.focal_length.x_y
    cx, cy = robot.camera.config.center.x_y
    camera_settings = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float)

    ###################
    at_goal = False
    picked_up = False

    count = 0

    """                 This requires a file to be in the same directory, functionality was demoed               """
    #angry_img = cozmo.oled_face.convert_image_to_screen_data(Image.open("angery.jpg").resize(cozmo.oled_face.dimensions(), Image.NEAREST))

    while (True):
        #robot.enable_stop_on_cliff(False)
        await robot.say_text("").wait_for_completed()
        if (robot.is_picked_up):
            if (not picked_up):
                at_goal = False
                pf.particles = Particle.create_random(PARTICLE_COUNT, grid)
                #await robot.display_oled_face_image(angry_img, 3000.0).wait_for_completed()
                await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
                picked_up = True
            continue

        if (at_goal):
            continue

        if (not flag_odom_init):
            last_pose = robot.pose
            flag_odom_init = True
            continue

        picked_up = False

        curr_odom = compute_odometry(robot.pose)
        last_pose = robot.pose

        markers, camera_image = await marker_processing(robot, camera_settings, show_diagnostic_image=False)

        m_x, m_y, m_h, m_confident = pf.update(curr_odom, markers)

        #robot.pose = cozmo.util.Pose(m_x,m_y,0,angle_z=cozmo.util.Angle(degrees=m_h))

        gui.show_particles(pf.particles)
        gui.show_mean(0,0,0)
        gui.show_camera_image(camera_image)
        gui.updated.set()

        count += 1

        if (m_confident):
            #robot.go_to_pose(cozmo.util.Pose(goal[0], goal[1], 0, angle_z=cozmo.util.Angle(degrees=goal[2])))
            robot.stop_all_motors()
            dx = goal[0] - m_x
            dy = goal[1] - m_y
            angle = math.atan2(dy,dx)
            distance = grid_distance(dx,dy,0,0)

            print(diff_heading_deg(math.degrees(angle), m_h))

            await robot.turn_in_place(degrees(diff_heading_deg(math.degrees(angle), m_h))).wait_for_completed()
            robot.wait_for_all_actions_completed()
            await robot.drive_straight(distance_inches(distance), speed_mmps(140)).wait_for_completed()
            robot.wait_for_all_actions_completed()
            await robot.turn_in_place(degrees(-1*diff_heading_deg(-5,math.degrees(angle)))).wait_for_completed()


            at_goal = True

            print("Done")
        else:
            """
            if (len(markers) > 0):
                await robot.turn_in_place(degrees(5)).wait_for_completed()
            else:
                await robot.turn_in_place(degrees(20)).wait_for_completed()

            if (count % 20 == 0):
                robot.stop_all_motors()
                await robot.drive_straight(distance_inches(3), speed_mmps(100)).wait_for_completed()
            """
            if (count % 5 == 0):
                await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()

            if (count > 30 and len(markers) > 0):
                robot.stop_all_motors()
                await robot.drive_straight(distance_inches(markers[0][0] * 0.5), speed_mmps(100)).wait_for_completed()
                await robot.turn_in_place(degrees(100)).wait_for_completed()
                await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()
                count -= 30

            """
            if (len(markers) > 0):
                robot.drive_wheel_motors(-10,10,-1000,1000)
            else:
                robot.drive_wheel_motors(-20,20,-1000,1000)
            """
            if (len(markers) > 0):
                await robot.turn_in_place(degrees(10)).wait_for_completed()
            else:
                await robot.turn_in_place(degrees(20)).wait_for_completed()


    ###################

class CozmoThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self, daemon=False)

    def run(self):
        cozmo.robot.Robot.drive_off_charger_on_connect = False  # Cozmo can stay on his charger
        cozmo.run_program(run, use_viewer=False)


if __name__ == '__main__':

    # cozmo thread
    cozmo_thread = CozmoThread()
    cozmo_thread.start()

    # init
    gui.show_particles(pf.particles)
    gui.show_mean(0, 0, 0)
    gui.start()
