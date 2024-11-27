import pinocchio as pin
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from messages_fr3.srv import JointConfig

class ManipulabilityMaximizer:
    def __init__(self, urdf_path, ee_frame_name, desired_translation, desired_rotation_euler, force_direction):
        # Load the robot model and create data for kinematics/dynamics calculations
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId(ee_frame_name)
        
        self.joint_config_publisher = self.create_publisher(JointConfig, '/joint_config', 10)

        self.subscription = self.create_subscription(JointConfig, '/joint_config', self.joint_config_callback, 10)

        # Set desired pose in SE(3)
        desired_rotation_matrix = R.from_euler('xyz', desired_rotation_euler).as_matrix()
        self.desired_pose = pin.SE3(desired_rotation_matrix, desired_translation)
        self.f_d = force_direction

    def objective_function(self, q):
        # Forward kinematics and Jacobian calculation
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
        pin.computeJointJacobians(self.model, self.data, q)
        
        # Retrieve the Jacobian in the base frame
        J = pin.getFrameJacobian(self.model, self.data, self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        
        # Objective function value calculation
        return -np.dot(self.f_d, np.linalg.inv(J @ J.T) @ self.f_d)  # Minimize the negative to maximize the value

    def pose_constraint(self, q):
        # Calculate forward kinematics for pose constraint
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
        
        # Calculate position and orientation error
        current_pose = self.data.oMf[self.ee_frame_id]
        position_error = current_pose.translation - self.desired_pose.translation
        orientation_error = pin.log3(self.desired_pose.rotation.T @ current_pose.rotation)
        
        # Combine errors into a single vector
        return np.concatenate([position_error, orientation_error])

    def optimize(self, q0, joint_limits_lower, joint_limits_upper, max_iterations=3000):
        # Define bounds from joint limits
        bounds = [(low, high) for low, high in zip(joint_limits_lower, joint_limits_upper)]
        constraints = {'type': 'eq', 'fun': self.pose_constraint}

        # Initial objective function evaluation
        initial_objective = self.objective_function(q0)
        print("Initial objective is:", initial_objective)

        # Run optimization
        result = minimize(
            self.objective_function, 
            q0, 
            constraints=constraints,
            bounds=bounds,
            method='SLSQP',  
            options={'maxiter': max_iterations, 'disp': True}
        )
        
        # Process optimization results
        if result.success:
            print("Optimal joint configuration:", result.x)
            print("Optimal joint configuration in degrees:", result.x * 180 / np.pi)
            
            # Extract optimal end-effector pose
            optimal_ee_pose = self.data.oMf[self.ee_frame_id]
            position = optimal_ee_pose.translation
            rotation_matrix = optimal_ee_pose.rotation
            euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz')
            
            print("Optimal EE position:", position)
            print("Optimal EE orientation (Euler angles):", euler_angles)
            print("Maximized objective value:", -result.fun)
        else:
            print("Optimization failed:", result.message)
        
        return result
    
    def joint_config_callback(self, request, response):
        # Extract desired pose and force direction from the service request
        desired_rotation_matrix = R.from_euler('xyz', [request.roll, request.pitch, request.yaw]).as_matrix()
        self.desired_pose = pin.SE3(desired_rotation_matrix, np.array([request.x, request.y, request.z]))
        self.f_d = np.array([request.directionx, request.directiony, request.directionz])



# Define parameters
urdf_path = '/home/nilsjohnson/franka_ros2_ws/src/cartesian_impedance_control/fr3.urdf'
ee_frame_name = 'fr3_hand'

# Initialize the optimizer class
optimizer = ManipulabilityMaximizer(urdf_path, ee_frame_name)

# Define joint limits and initial configuration
joint_limits_lower = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159, -0.04, -0.04])
joint_limits_upper = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159, 0.04, 0.04])
q0 = np.array([0.33506416, 0.19438218, -0.34938642, -2.4187224, 1.59775894, 1.5097755, -1.34924279, 0., 0.])

# Run the optimization
optimizer.optimize(q0, joint_limits_lower, joint_limits_upper)
