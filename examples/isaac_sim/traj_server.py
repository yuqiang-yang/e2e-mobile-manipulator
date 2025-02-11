import torch
import time
import matplotlib.pyplot as plt

class TrajServer:
    def __init__(self):
        self.trajectory = None
        self.time_stamps = None
        self.writing = False
    def ready(self):
        return self.trajectory is not None and self.time_stamps is not None
    def set_trajectory(self, trajectory, time_stamps):
        """
        Set the trajectory tensor and its corresponding time stamps.
        :param trajectory: Tensor of shape (batch, num_steps, num_joints)
        :param time_stamps: Tensor of shape (num_steps,) containing the absolute time for each trajectory point
        """
        if trajectory.shape[1] != time_stamps.shape[0]:
            raise ValueError("Trajectory and time_stamps must have the same number of steps")
        self.trajectory = trajectory
        self.time_stamps = time_stamps

    def get_robot_state(self, cur_time, batch_idx=None):
        """
        Get the robot state at the given time by interpolating the trajectory.
        :param cur_time: Absolute time at which to get the robot state
        :param batch_idx: Index of the batch to get the robot state from
        :return: Interpolated robot state as a Tensor
        """
        while self.writing:
            time.sleep(0.001)
        if self.trajectory is None or self.time_stamps is None:
            raise ValueError("Trajectory or time_stamps are not set")

        if cur_time < self.time_stamps[0] or cur_time > self.time_stamps[-1]:
            # print(f"Time {time} is out of bounds ({self.time_stamps[0]}, {self.time_stamps[-1]})")
            pass

        # Binary search to find the two nearest time steps
        idx = torch.searchsorted(self.time_stamps, cur_time).item()
        if idx == 0:
            return self.trajectory[batch_idx, 0] if batch_idx is not None else self.trajectory[:, 0]
        if idx >= self.time_stamps.shape[0]:
            return self.trajectory[batch_idx, -1] if batch_idx is not None else self.trajectory[:, -1]

        t1 = self.time_stamps[idx - 1]
        t2 = self.time_stamps[idx]

        # Linear interpolation
        alpha = (cur_time - t1) / (t2 - t1)
        try:
            if batch_idx is None:
                state = (1 - alpha) * self.trajectory[:, idx - 1] + alpha * self.trajectory[:, idx]
            else:
                state = (1 - alpha) * self.trajectory[batch_idx, idx - 1] + alpha * self.trajectory[batch_idx, idx]
        except:

            import ipdb; ipdb.set_trace()
        return state

    def trim_sub_trajectory(self, lower_bound, upper_bound):
        """
        Trim the trajectory to only include points within the given time bounds.
        :param lower_bound: The lower bound of the time range
        :param upper_bound: The upper bound of the time range
        """
        if self.trajectory is None or self.time_stamps is None:
            print("[Warn][TrajServer] try to trim an empty trajectory")
            return 

        mask = (self.time_stamps >= lower_bound) & (self.time_stamps <= upper_bound)
        self.trajectory = self.trajectory[:, mask]
        self.time_stamps = self.time_stamps[mask]

    def concat_trajectory(self, new_trajectory, new_time_stamps):
        """
        Concatenate a new trajectory and its time stamps to the existing trajectory.
        :param new_trajectory: Tensor of shape (batch, num_steps, num_joints) for the new trajectory
        :param new_time_stamps: Tensor of shape (num_steps,) containing the absolute time for each new trajectory point
        """
        if new_trajectory.shape[1] != new_time_stamps.shape[0]:
            raise ValueError("New trajectory and new time_stamps must have the same number of steps")

        if self.trajectory is None or self.time_stamps is None:
            self.trajectory = new_trajectory
            self.time_stamps = new_time_stamps
        else:
            self.trajectory = torch.cat((self.trajectory, new_trajectory), dim=1)
            self.time_stamps = torch.cat((self.time_stamps, new_time_stamps), dim=0)

    def plot_trajectory(self):
        if self.trajectory is None or self.time_stamps is None:
            raise ValueError("Trajectory or time_stamps are not set")

        # 获取第一个批次的轨迹
        trajectory = self.trajectory[0].cpu().numpy()
        time_stamps = self.time_stamps

        # 提取前四维的位置和速度
        positions = trajectory[:, 1:4]
        velocities = trajectory[:, 13:16]

        # 启用交互模式
        plt.ion()

        # 创建第一个窗口，绘制位置
        plt.figure(1)
        plt.clf()
        for i in range(3):
            plt.plot(time_stamps, positions[:, i], label=f'Position {i+1}')
        plt.xlabel('Time')
        plt.ylabel('Position')
        plt.title('Positions of the first batch')
        plt.legend()
        plt.grid(True)
        plt.draw()
        plt.pause(0.001)

        # 创建第二个窗口，绘制速度
        plt.figure(2)
        plt.clf()
        for i in range(3):
            plt.plot(time_stamps, velocities[:, i], label=f'Velocity {i+1}')
        plt.xlabel('Time')
        plt.ylabel('Velocity')
        plt.title('Velocities of the first batch')
        plt.legend()
        plt.grid(True)
        plt.draw()
        plt.pause(0.001)

def shift_seeds(tensor, idx):
    """
    Shift the elements of the second dimension of the tensor to the front by idx positions.
    The last idx elements are filled with the last element's value.
    
    :param tensor: Tensor of shape (batch, num_steps, num_features)
    :param idx: Number of positions to shift
    :return: Shifted tensor
    """
    if idx < 0 or idx >= tensor.shape[1]:
        raise ValueError("idx must be in the range [0, num_steps)")

    # Get the shape of the tensor
    batch_size, num_steps, num_features = tensor.shape

    # Create a new tensor to store the shifted values
    shifted_tensor = torch.empty_like(tensor)

    # Shift the elements
    shifted_tensor[:, :-idx, :] = tensor[:, idx:, :]

    # Fill the last idx elements with the last element's value
    shifted_tensor[:, -idx:, :] = tensor[:, -idx:, :].expand(batch_size, idx, num_features)

    return shifted_tensor

if __name__ == "__main__":
    traj_server = TrajServer()

    # Example trajectory: 2 batches, 10 steps, 6 joints
    trajectory = torch.rand((2, 10, 6))
    time_stamps = torch.linspace(0, 9, steps=10)  # Example time stamps

    traj_server.set_trajectory(trajectory, time_stamps)

    # Trim trajectory
    traj_server.trim_sub_trajectory(2, 7)
    print(f"Trimmed trajectory: {traj_server.trajectory}")
    print(f"Trimmed time stamps: {traj_server.time_stamps}")

    # Concatenate new trajectory
    new_trajectory = torch.rand((2, 5, 6))
    new_time_stamps = torch.linspace(10, 14, steps=5)
    traj_server.concat_trajectory(new_trajectory, new_time_stamps)
    print(f"Concatenated trajectory: {traj_server.trajectory}")
    print(f"Concatenated time stamps: {traj_server.time_stamps}")

    # Get robot state
    time = 3.5  # Absolute time
    batch_idx = None  # Batch index
    state = traj_server.get_robot_state(time, batch_idx)
    print(f"Robot state at time {time} for batch {batch_idx}: {state}")