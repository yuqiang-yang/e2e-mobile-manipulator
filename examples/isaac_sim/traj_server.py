import torch

class TrajServer:
    def __init__(self):
        self.trajectory = None
        self.time_stamps = None
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

    def get_robot_state(self, time, batch_idx=None):
        """
        Get the robot state at the given time by interpolating the trajectory.
        :param time: Absolute time at which to get the robot state
        :param batch_idx: Index of the batch to get the robot state from
        :return: Interpolated robot state as a Tensor
        """
        if self.trajectory is None or self.time_stamps is None:
            raise ValueError("Trajectory or time_stamps are not set")

        if time < self.time_stamps[0] or time > self.time_stamps[-1]:
            # print(f"Time {time} is out of bounds ({self.time_stamps[0]}, {self.time_stamps[-1]})")
            pass

        # Binary search to find the two nearest time steps
        idx = torch.searchsorted(self.time_stamps, time).item()
        if idx == 0:
            return self.trajectory[batch_idx, 0] if batch_idx is not None else self.trajectory[:, 0]
        if idx >= self.time_stamps.shape[0]:
            return self.trajectory[batch_idx, -1] if batch_idx is not None else self.trajectory[:, -1]

        t1 = self.time_stamps[idx - 1]
        t2 = self.time_stamps[idx]

        # Linear interpolation
        alpha = (time - t1) / (t2 - t1)
        if batch_idx is None:
            state = (1 - alpha) * self.trajectory[:, idx - 1] + alpha * self.trajectory[:, idx]
        else:
            state = (1 - alpha) * self.trajectory[batch_idx, idx - 1] + alpha * self.trajectory[batch_idx, idx]

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

# Example usage
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