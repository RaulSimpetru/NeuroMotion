import opensim as osim
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from NeuroMotion.MSKlib.pose_params import RANGE_DOF


class MSKModel:
    def __init__(
        self,
        model_path="NeuroMotion/MSKlib/models/ARMS_Wrist_Hand_Model_4.3",
        model_name="Hand_Wrist_Model_for_development.osim",
        default_pose_path="NeuroMotion/MSKlib/models/poses.csv",
    ):

        self.mov = []

        self._init_msk_model(str(Path(model_path, model_name).resolve()))
        self._init_pose(Path(default_pose_path).resolve())

        self.muscle_lengths = None

    def _init_msk_model(self, model_path):
        # Load the model
        self.model = osim.Model(model_path)

        # Get all muscles
        self.all_ms_labels = [ms.getName() for ms in self.model.getMuscles()]

    def _init_pose(self, pose_path):
        self.pose_basis = pd.read_csv(pose_path)

    def _check_range(self):
        if len(self.mov) > 0:
            for k, v in RANGE_DOF.items():
                if k in self.mov:
                    dof = (
                        self.mov[k] * np.pi / 180
                        if k in {"deviation", "flexion"}
                        else self.mov[k]
                    )
                    assert np.all(dof >= v[0]) and np.all(
                        dof <= v[1]
                    ), f"DoF {k} out of range!"
        else:
            print("mov has not been initialized!")

    def load_mov(self, angles):
        """
        angles: np.array or dataframe of the joint angles
        Requires:
        if angles is np.array, the first column of angles should be the time in seconds. The second to the 25th columns are the angles of the 24 DoFs, in the same sequence with pose_basis
        if angles is pd.dataframe, the columns should be 'time' and the DoFs in pose_basis
        """
        num_columns = len(self.pose_basis) + 1
        if isinstance(angles, np.ndarray):
            assert (
                angles.shape[1] == num_columns
            ), f"input angle with {angles.shape[1]} columns, required {num_columns} columns"
            self.mov = pd.DataFrame(
                data=angles, columns=["time", *self.pose_basis.iloc[:, 0].tolist()]
            )
        elif isinstance(angles, pd.DataFrame):
            assert (
                len(angles.columns) == num_columns
            ), f"input angle with {len(angles.columns)} columns, required {num_columns} columns"
            self.mov = angles
        else:
            raise NotImplementedError(
                "Not implemented angle type. Should be np.array or pd.dataframe."
            )

        self._check_range()

    def update_mov(self, mov):
        self.mov = mov
        self._check_range()

    def sim_mov(
        self,
        movement_sampling_frequency: int,
        pose_sequence: list[str],
        pose_transition_times: list[float],
    ):
        try:
            assert len(pose_sequence) - 1 == len(pose_transition_times)
        except AssertionError:
            print(
                "The number of poses does not match the number of transition times. "
                "There should be one less transition time than the number of poses."
            )

        # Get pd of time and joints
        mov = []
        total_time_dim = 0

        for pose_transition_time, (current_pose_name, next_pose_name) in zip(
            pose_transition_times, zip(pose_sequence[:-1], pose_sequence[1:])
        ):
            time_dim = int(pose_transition_time * movement_sampling_frequency)

            current_pose = current_pose_name.replace("default", "open").split("+")
            next_pose = next_pose_name.replace("default", "open").split("+")

            current_angles = self.pose_basis[current_pose].sum(axis=1).values
            next_angles = self.pose_basis[next_pose].sum(axis=1).values

            mov.append(np.linspace(current_angles, next_angles, num=time_dim))
            total_time_dim = total_time_dim + time_dim

        self.mov = pd.DataFrame(
            data=np.concatenate(
                (
                    np.linspace(0, np.sum(pose_transition_times), num=total_time_dim)[
                        :, None
                    ],
                    np.concatenate(mov),
                ),
                axis=1,
            ),
            columns=["time", *self.pose_basis.iloc[:, 0].tolist()],
        )

        self._check_range()

    def write_mov(self, res_path):
        mov_fl = self.mov.copy(deep=True)
        # Only these two DoFs should be converted to radian
        mov_fl.loc[:, "deviation"] = mov_fl.loc[:, "deviation"] * np.pi / 180
        mov_fl.loc[:, "flexion"] = mov_fl.loc[:, "flexion"] * np.pi / 180
        mov_fl.to_csv(res_path, sep="\t", index=False)
        header = (
            "motionfile\n"
            + "version=1\n"
            + "nRows={}\n".format(mov_fl.shape[0])
            + "nColumns={}\n".format(mov_fl.shape[1])
            + "inDegrees=yes\n"
            + "endheader\n"
            + "\n"
        )
        with open(res_path, "r+") as f:
            content = f.read()
            f.seek(0, 0)
            f.write(header + content)

    def mov2len(
        self,
        muscle_labels: list[str],
        normalise: bool = True,
        default_pose_label: str = "open",
    ):
        state = self.model.initSystem()

        # Get default muscle length for normalisation
        for dof, deg in zip(
            self.mov.columns[1:], self.pose_basis[default_pose_label].values
        ):
            self.model.updCoordinateSet().get(dof).setValue(
                state, np.radians(deg), False
            )
        self.model.assemble(state)
        self.model.equilibrateMuscles(state)

        default_muscle_lengths = np.array(
            [
                self.model.getMuscles().get(ms).getFiberLength(state)
                for ms in muscle_labels
            ]
        )

        # Run with time steps
        muscle_lengths = []
        for row in tqdm(
            self.mov.itertuples(index=False, name=None),
            desc="Extracting muscle lengths during movement of MSK model",
            total=len(self.mov),
        ):
            for dof, deg in zip(self.mov.columns[2:], row[1:]):
                self.model.updCoordinateSet().get(dof).setValue(state, np.radians(deg))
                self.model.realizePosition(state)
            self.model.equilibrateMuscles(state)

            muscle_lengths.append(
                [
                    self.model.getMuscles().get(ms).getFiberLength(state)
                    for ms in muscle_labels
                ]
            )

        muscle_lengths = np.array(muscle_lengths)

        if normalise:
            muscle_lengths = muscle_lengths / default_muscle_lengths[None, ...]

        self.muscle_lengths = pd.DataFrame(
            data=np.concatenate(
                (self.mov["time"].values[:, None], muscle_lengths), axis=1
            ),
            columns=["time", *muscle_labels],
        )

    def len2params(self):
        # Assumption: constant volume
        # If lens change by s, correspondingly depths will change by 1/sqrt(s) and cvs will change by 1/s.
        # The outputs are in normalised scales
        # Use it with a predefined absolute value between 0.5 and 1.0

        if self.muscle_lengths is None:
            raise ValueError(
                "muscle_lengths has not been initialised. Run mov2len first."
            )

        depths = self.muscle_lengths.copy(deep=True)
        depths.iloc[:, 1:] = 1 / (np.sqrt(depths.iloc[:, 1:]) + 1e-8)

        cvs = self.muscle_lengths.copy(deep=True)
        cvs.iloc[:, 1:] = 1 / (cvs.iloc[:, 1:] + 1e-8)

        param_changes = {
            "depth": depths,
            "cv": cvs,
            "len": self.muscle_lengths,
            "steps": len(depths),
        }

        self.param_changes = param_changes


if __name__ == "__main__":
    # test pose to mov
    msk = MSKModel()

    poses = ["default", "default+flex", "default", "default+ext", "default"]
    durations = [2] * 4
    fs = 5
    ms_labels = ["ECRB", "ECRL", "PL", "FCU", "ECU", "EDCI", "FDSI"]

    msk.sim_mov(fs, poses, durations)
    msk.write_mov("./res/mov.mot")
    ms_lens = msk.mov2len(muscle_labels=ms_labels)
    changes = msk.len2params()
