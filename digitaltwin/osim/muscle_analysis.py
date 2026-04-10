import numpy as np
import opensim as osim
import matplotlib.pyplot as plt
import pandas as pd
import time
import json
import sys
import os
import io

from digitaltwin.utils.array_tools import find_nearest_idx
from contextlib import redirect_stdout


def open_json(file_path):
    with open(file_path, 'r') as file:
        json_data = json.load(file)

    parameter = {}
    parameter.update({"people": json_data['people']})
    parameter.update({"sport": json_data['sport']})
    parameter.update({"label": json_data['label']})
    parameter.update({"folder": json_data['folder']})
    parameter.update({"model_folder": json_data['model_folder']})
    parameter.update({"model_file": os.path.join(json_data['model_folder'], json_data['model_file'])})
    parameter.update({"motion_file": os.path.join(json_data['folder'], json_data['motion_file'])})
    current_directory = os.getcwd()
    parameter.update({"result_dir": os.path.join(current_directory, json_data['result_dir'])})
    if 'robot_file' in json_data:
        parameter.update({"robot_file": os.path.join(json_data['folder'], json_data['robot_file'])})
    if 'robot_time_delay' in json_data:
        parameter.update({"robot_time_delay": json_data['robot_time_delay']})
    if 'force_file' in json_data:
        parameter.update({"force_file": os.path.join(json_data['folder'], json_data['force_file'])})
    if 'external_load_file' in json_data:
        file_name = f"Setup_ExternalForce_{parameter['label']}.xml"
        parameter.update({"external_load_file": os.path.join(parameter['result_dir'], file_name)})
    if 'target_inverse_dynamics' in json_data:
        file_name = f"Setup_InverseDynamics_{parameter['label']}.xml"
        parameter.update({"target_inverse_dynamics": os.path.join(parameter['result_dir'], file_name)})
    if 'target_muscle_analysis' in json_data:
        file_name = f"Setup_MuscleAnalysis_{parameter['label']}.xml"
        parameter.update({"target_muscle_analysis": os.path.join(parameter['result_dir'], file_name)})
    return parameter


def print_model_information(model_file):
    target_model = osim.Model(model_file)

    # Print metadata.
    print("Name of the model:", target_model.getName())
    print("Author:", target_model.get_credits())
    print("Publications:", target_model.get_publications())
    print("Length Unit:", target_model.get_length_units())
    print("Force Unit:", target_model.get_force_units())
    print("Gravity:", target_model.get_gravity())

    # Print the number of coordinates.
    print("Num Coordinates:", target_model.getNumCoordinates())
    print()

    # For each coordinate, print some information, such as its name or motion type.
    for coordinate in target_model.getCoordinateSet():
        print("  Coordinate Name:", coordinate.getName())
        print("  Coordinate Absolute Path:", coordinate.getAbsolutePathString())

        # Motion type is an enumerate (0:Undefined, 1:Rotational, 2:Translational, 3:Coupled).
        motion_type = coordinate.getMotionType()
        motion_type_string = ""
        if motion_type == 0:
            motion_type_string = "Undefined"
        elif motion_type == 1:
            motion_type_string = "Rotational"
        elif motion_type == 2:
            motion_type_string = "Translational"
        elif motion_type == 3:
            motion_type_string = "Coupled"
        print("  Coordinate Motion Type:", motion_type_string)

        print()

    # Print number of joints.
    print("Num Joints:", target_model.getNumJoints())
    print()

    # For each joint, print some information, such as its name or components.
    for joint in target_model.getJointSet():
        print("Joint Name:", joint.getName())
        print("Joint Absolute Path:", joint.getAbsolutePathString())
        print("Components:")
        for component in joint.getComponentsList():
            print("  Component Name:", component.getName())
            print("  Component Absolute Path:", component.getAbsolutePathString())
        print()

    # Print the number of muscles.
    print("Num Muscles:", target_model.getMuscles().getSize())
    print()

    for muscle in target_model.getMuscles():
        print("Muscle Name:", muscle.getName())
        print("Muscle Absolute Path:", muscle.getAbsolutePathString())
        print()


def plot_motion_file(file_name):
    # Use the TableProcessor to read the motion file.
    tableTime = osim.TimeSeriesTable(file_name)
    # Process the file.
    # Print the column labels.
    print(tableTime.getColumnLabels())

    # Get columns we want to analyze, and the time column (independent column).
    y_knee_angle_r = tableTime.getDependentColumn('knee_angle_r')
    y_knee_angle_l = tableTime.getDependentColumn('knee_angle_l')
    y_hip_adduction_r = tableTime.getDependentColumn('hip_adduction_r')
    y_hip_adduction_l = tableTime.getDependentColumn('hip_adduction_l')
    x_time = tableTime.getIndependentColumn()

    # Create six subplots, with 2 rows and 3 columns.
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle('Normal Gait')

    # Plot the knee angles on the first subplot.
    axs[0].plot(x_time, y_knee_angle_r.to_numpy(), label='knee_angle_r')
    axs[0].plot(x_time, y_knee_angle_l.to_numpy(), label='knee_angle_l')
    axs[0].set_title('Knee angle')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Knee angle')
    axs[0].grid()
    axs[0].legend()

    # Plot the hip adductions on the second subplot.
    axs[1].plot(x_time, y_hip_adduction_r.to_numpy(), label='hip_adduction_r')
    axs[1].plot(x_time, y_hip_adduction_l.to_numpy(), label='hip_adduction_l')
    axs[1].set_title('Hip adduction')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Hip Adduction')
    axs[1].grid()
    axs[1].legend()

    # Set the spacing between subplots
    plt.subplots_adjust(wspace=0.5, hspace=0.5)


def range_of_motion():
    # Use the TableProcessor to read the motion file.
    table_processor_normal_gait = osim.TableProcessor('normal_gait.mot')
    # Process the file.
    table_normal_gait = table_processor_normal_gait.process()
    # Print labels for each column.
    print(table_normal_gait.getColumnLabels())

    # Use the TableProcessor to read the motion file.
    table_processor_crouch_gait = osim.TableProcessor('crouch_gait.mot')
    # Process the file.
    table_crouch_gait = table_processor_crouch_gait.process()
    # Print labels for each column.
    print(table_crouch_gait.getColumnLabels())

    # Get columns we want to represent.
    normal_gait_knee_angle_r = table_normal_gait.getDependentColumn('knee_angle_r')
    crouch_gait_knee_angle_r = table_crouch_gait.getDependentColumn('knee_angle_r')

    # Get independent columns of each table (time).
    normal_gait_time = table_normal_gait.getIndependentColumn()
    crouch_gait_time = table_crouch_gait.getIndependentColumn()

    # Create six subplots, with 2 rows and 3 columns.
    fig, axs = plt.subplots(1, 1, figsize=(7, 5))
    fig.suptitle('Normal Gait vs. Crouch Gait')

    # Plot the knee angles on the first subplot.
    axs.plot(normal_gait_time, normal_gait_knee_angle_r.to_numpy(), label='normal_gait_knee_angle_r')
    axs.plot(crouch_gait_time, crouch_gait_knee_angle_r.to_numpy(), label='crouch_gait_knee_angle_l')
    axs.set_title('Knee angle')
    axs.set_xlabel('Time')
    axs.set_ylabel('Knee angle')
    axs.grid()
    axs.legend()

    # Set the spacing between subplots
    plt.subplots_adjust(wspace=0.5, hspace=0.5)


def monotonically_increasing_with_threshold(arr, arrt, threshold1=None, threshold2=None):
    arr = np.asarray(arr)
    arrt = np.asarray(arrt)
    if threshold1 is None:
        threshold1 = min(arr)
    if threshold2 is None:
        threshold2 = max(arr)
    diff = arr[1:] - arr[:-1]
    # diff = np.where(diff > 0, diff, np.zeros_like(diff))
    begin_flag = False
    begin_index = 0
    for i in range(diff.shape[0]):
        if begin_flag is False and diff[i] > 0 and threshold1 < arr[i] < threshold2:
            begin_flag = True
            begin_index = i
        if begin_flag is True and diff[i] <= 0:
            begin_flag = False
            if i - begin_index > 10:
                print(begin_index + 1, '-', i + 1, '\t',
                      "{:.3f}, {:.3f}".format(arrt[begin_index], arrt[i]),
                      "({:.2f}s)".format(arrt[i] - arrt[begin_index]), '\t',
                      arr[begin_index], '°', '-', arr[i], '°')
        if begin_flag is True and arr[i] >= threshold2:
            begin_flag = False
            if i - begin_index > 10:
                print(begin_index + 1, '-', i + 1, '\t',
                      "{:.3f}, {:.3f}".format(arrt[begin_index], arrt[i]),
                      "({:.2f}s)".format(arrt[i] - arrt[begin_index]), '\t',
                      arr[begin_index], '°', '-', arr[i], '°')


def monotonically_decreasing_with_threshold(arr, arrt, threshold1=None, threshold2=None):
    arr = np.asarray(arr)
    arrt = np.asarray(arrt)
    if threshold1 is None:
        threshold1 = min(arr)
    if threshold2 is None:
        threshold2 = max(arr)
    diff = arr[1:] - arr[:-1]
    diff = np.where(diff < 0, diff, np.zeros_like(diff) * (-1))
    begin_flag = False
    begin_index = 0
    for i in range(diff.shape[0]):
        if begin_flag is False and diff[i] < 0 and threshold1 < arr[i] < threshold2:
            begin_flag = True
            begin_index = i
        if begin_flag is True and diff[i] >= 0:
            begin_flag = False
            if i - begin_index > 10:
                print(begin_index + 1, '-', i + 1, '\t',
                      "{:.3f}, {:.3f}".format(arrt[begin_index], arrt[i]),
                      "({:.2f}s)".format(arrt[i] - arrt[begin_index]), '\t',
                      arr[begin_index], '°', '-', arr[i], '°')
        if begin_flag is True and threshold1 >= arr[i]:
            begin_flag = False
            if i - begin_index > 10:
                print(begin_index + 1, '-', i + 1, '\t',
                      "{:.3f}, {:.3f}".format(arrt[begin_index], arrt[i]),
                      "({:.2f}s)".format(arrt[i] - arrt[begin_index]), '\t',
                      arr[begin_index], '°', '-', arr[i], '°')


def generate_force_mot(parameter):
    print("-" * 50, "Generating file of external force.", "-" * 50)

    robot_file = parameter['robot_file']
    robot_data = pd.read_csv(robot_file, skiprows=18)

    hand_l_file = os.path.join(parameter['result_dir'],
                               f"{parameter['label']}_{parameter['sport']}_PointKinematics_hand_l_pos.sto")
    hand_r_file = os.path.join(parameter['result_dir'],
                               f"{parameter['label']}_{parameter['sport']}_PointKinematics_hand_r_pos.sto")
    hand_l_pos = pd.read_csv(hand_l_file, sep='\t', skiprows=7)
    hand_r_pos = pd.read_csv(hand_r_file, sep='\t', skiprows=7)

    def plot_time_delay(hand_l_pos, robot_data):
        plt.figure(figsize=(8.7, 3.7))
        plt.subplots_adjust(left=0.09, bottom=0.1868, right=0.955, top=0.862)
        plt.plot(hand_l_pos['time'], hand_l_pos['barbell_l_pz'] + 0.6, label='Xsens')
        plt.plot(robot_data['time'], robot_data['barbell_pos_l'], label='robot')
        # plt.plot(robot_data[' time'] - 54.5, robot_data[' axis4_pos(m)'] - 0.13, label='robot')
        # plt.xlim([20, 35])
        # plt.xlim([10, 35])
        # plt.xlim([10, 40])
        plt.xlim([5, 40])
        plt.xlabel('s')
        plt.ylabel('m')
        plt.legend()
        plt.savefig(parameter['result_dir'] + f"/result_{parameter['label']}.png")
        # plt.show()

    # plot_time_delay(hand_l_pos, robot_data)
    hand_l_pos = hand_l_pos[['time', 'state_0', 'state_1', 'state_2']].rename(columns={
        'state_0': 'barbell_l_px',
        'state_1': 'barbell_l_pz',
        'state_2': 'barbell_l_py'
    })
    hand_r_pos = hand_r_pos[['time', 'state_0', 'state_1', 'state_2']].rename(columns={
        'state_0': 'barbell_r_px',
        'state_1': 'barbell_r_pz',
        'state_2': 'barbell_r_py'
    })
    robot_force = robot_data[[' time', ' axis3_force(N)', ' axis4_force(N)', ' axis4_pos(m)']].rename(columns={
        ' time': 'time',
        ' axis3_force(N)': 'barbell_r_vz',
        ' axis4_force(N)': 'barbell_l_vz',
        ' axis4_pos(m)': 'barbell_pos_l'
    })
    robot_force['barbell_pos_l'] = 0.7 - robot_force['barbell_pos_l']
    assert hand_l_pos['time'].all() == hand_r_pos['time'].all()
    hand_pos = pd.concat([hand_l_pos, hand_r_pos.drop(columns='time')], axis=1, ignore_index=False)

    robot_idx = []
    robot_force['time'] = robot_force['time'] - parameter['robot_time_delay']
    for i in range(len(hand_pos['time'])):
        j = find_nearest_idx(robot_force['time'], hand_pos['time'][i])
        robot_idx.append(j)
    assert len(robot_idx) == len(hand_pos['time'])
    force_data = robot_force.iloc[robot_idx]

    plot_time_delay(hand_l_pos, force_data)
    print_buffer = io.StringIO()
    with open(parameter['result_dir'] + f"/result_{parameter['label']}.txt", 'a') as f, redirect_stdout(print_buffer):
        monotonically_increasing_with_threshold(force_data['barbell_pos_l'], force_data['time'])
        monotonically_decreasing_with_threshold(force_data['barbell_pos_l'], force_data['time'])
        f.write(print_buffer.getvalue())
    print(print_buffer.getvalue())

    force_data = force_data.drop(columns=['time', 'barbell_pos_l'])
    force_data['barbell_r_vx'] = 0
    force_data['barbell_r_vy'] = 0
    force_data['barbell_l_vx'] = 0
    force_data['barbell_l_vy'] = 0
    force_data = force_data.reset_index(drop=True)
    merged_data = pd.concat([hand_pos, force_data], axis=1, ignore_index=False)

    # Generate '.mot' file header.
    mot_file_content = []
    force_file_name = f"{parameter['label']}_{parameter['sport']}_force.mot"
    mot_file_content.append("barbell reaction force")
    mot_file_content.append(f"nRows={len(merged_data)}")
    mot_file_content.append(f"nColumns={len(merged_data.columns)}")
    mot_file_content.append("inDegrees=no")
    mot_file_content.append("# OpenSim Motion File Header")
    mot_file_content.append(f"name {force_file_name}")
    mot_file_content.append(f"datacolumns {len(merged_data.columns)}")
    mot_file_content.append(f"datarows {len(merged_data)}")
    mot_file_content.append("otherdata 1")
    mot_file_content.append("endheader")
    mot_file_content.append("")

    # Add data.
    mot_file_content.append("\t".join(merged_data.columns))  # 添加列名
    for index, row in merged_data.iterrows():
        mot_file_content.append("\t".join(map(str, row.values)))

    # Save '.mot' file.
    with open(os.path.join(parameter['result_dir'], force_file_name), 'w') as f:
        f.write("\n".join(mot_file_content))
    parameter.update({"force_file": os.path.join(parameter['result_dir'], force_file_name)})
    return parameter


def point_kinematics(parameter, label='hand_l'):
    model_file = parameter['model_file']
    motion_file = parameter['motion_file']
    result_dir = parameter['result_dir']
    target_xml = result_dir + "/Setup_PointKinematics_" + parameter['label'] + '_' + label + ".xml"
    print("-" * 50, "Calculating point kinematics of", label, "-" * 50)

    # Load motion file and extract time.
    tableTime = osim.TimeSeriesTable(motion_file)
    x_time = tableTime.getIndependentColumn()

    # Get first and last time.
    first_time = x_time[0]
    last_time = x_time[len(x_time) - 1]

    # Define a MuscleAnalysis.
    point_kinematics = osim.PointKinematics()

    # Set start and end times for the analysis.
    point_kinematics.setStartTime(first_time)
    point_kinematics.setEndTime(last_time)

    target_model = osim.Model(model_file)
    body_ground = target_model.getGround()
    body_part = target_model.getBodySet().get(label)

    point_kinematics.setRelativeToBody(body_ground)
    point_kinematics.setBody(body_part)

    # Configure the analysis.
    point_kinematics.setOn(True)
    point_kinematics.setStepInterval(1)
    point_kinematics.setInDegrees(True)
    point_kinematics.setPointName(label)

    # Create an AnalyzeTool for normal gait.
    analyze_tool = osim.AnalyzeTool()
    analyze_tool.setName(parameter['label'] + '_' + parameter['sport'])

    # Set the model file and motion file to analyze.
    analyze_tool.setModelFilename(model_file)
    analyze_tool.setCoordinatesFileName(motion_file)

    # Add the MuscleAnalysis to the AnalyzeTool.
    analyze_tool.updAnalysisSet().cloneAndAppend(point_kinematics)

    # Directory where results are stored.
    analyze_tool.setResultsDir(result_dir)

    # Configure AnalyzeTool.
    analyze_tool.setReplaceForceSet(False)
    analyze_tool.setSolveForEquilibrium(True)  # 待确定
    analyze_tool.setStartTime(first_time)
    analyze_tool.setFinalTime(last_time)

    # Print configuration of the AnalyzeTool to an XML file.
    analyze_tool.printToXML(target_xml)

    analyze_tool = osim.AnalyzeTool(target_xml, True)
    result = analyze_tool.run()


def generate_external_force_xml(name, force_file, target_file):
    def set_external_force(name, body_part, force_name, point_name):
        external_force = osim.ExternalForce()
        external_force.setName(name)
        external_force.set_applied_to_body(body_part)
        external_force.set_force_expressed_in_body("ground")
        external_force.set_point_expressed_in_body("ground")
        external_force.set_force_identifier(force_name)
        external_force.set_point_identifier(point_name)
        return external_force

    # force_ground_r = set_external_force("Right_GRF", "toes_r", "ground_force_r_v", "ground_force_r_p")
    # force_ground_l = set_external_force("Left_GRF", "toes_l", "ground_force_l_v", "ground_force_l_p")
    force_hand_r = set_external_force("Hand_r_RF", "hand_r", "barbell_r_v", "barbell_r_p")
    force_hand_l = set_external_force("Hand_l_RF", "hand_l", "barbell_l_v", "barbell_l_p")

    external_loads = osim.ExternalLoads()
    external_loads.setName(name)
    external_loads.setDataFileName(force_file)
    # external_loads.cloneAndAppend(force_ground_r)
    # external_loads.cloneAndAppend(force_ground_l)
    external_loads.cloneAndAppend(force_hand_r)
    external_loads.cloneAndAppend(force_hand_l)

    external_loads.printToXML(target_file)


def inverse_dynamics(parameter, name, model_file, motion_file, external_load_file, target_file=None):
    print("-" * 50, "Calculating inverse dynamics.", "-" * 50)

    # Define a InverseDynamicsTool.
    inverse_dynamics = osim.InverseDynamicsTool()
    inverse_dynamics.setName(name)
    result_dir = os.path.join(parameter['result_dir'], parameter['label'])
    inverse_dynamics.setResultsDir(result_dir)
    inverse_dynamics.setModelFileName(model_file)
    inverse_dynamics.setExternalLoadsFileName(external_load_file)
    inverse_dynamics.setCoordinatesFileName(motion_file)
    # inverse_dynamics.setLowpassCutoffFrequency(6)

    # Set time range.
    tableTime = osim.TimeSeriesTable(motion_file)
    x_time = tableTime.getIndependentColumn()
    first_time = x_time[0]
    last_time = x_time[len(x_time) - 1]
    inverse_dynamics.setStartTime(first_time)
    inverse_dynamics.setEndTime(last_time)

    ExcludedForce = osim.ArrayStr()
    ExcludedForce.append("Muscles")
    inverse_dynamics.setExcludedForces(ExcludedForce)

    if target_file is not None:
        inverse_dynamics.printToXML(target_file)

    # inverse_dynamics = osim.InverseDynamicsTool(target_file)
    inverse_dynamics.run()


def muscle_analysis(parameter, model_file, motion_file, external_load_file, target_xml=None):
    print("-" * 50, "Calculating msucle analysis.", "-" * 50)

    # Load motion file and extract time.
    tableTime = osim.TimeSeriesTable(motion_file)
    x_time = tableTime.getIndependentColumn()

    # Get first and last time.
    first_time = x_time[0]
    last_time = x_time[len(x_time) - 1]

    # Define a MuscleAnalysis.
    muscle_analysis = osim.MuscleAnalysis()

    # Set start and end times for the analysis.
    muscle_analysis.setStartTime(first_time)
    muscle_analysis.setEndTime(last_time)

    # Set the muscle of interest.
    muscle_list = osim.ArrayStr()
    muscle_list.append("all")
    # muscle_list.append("rect_abd_l")
    muscle_analysis.setMuscles(muscle_list)

    # Set the joint of interest.
    moment_list = osim.ArrayStr()
    # moment_list.append("flex_extension")
    # moment_list.append("hip_flexion_l")
    # moment_list.append("knee_angle_l")
    # moment_list.append("ankle_angle_l")
    moment_list.append("arm_flex_l")
    moment_list.append("arm_add_l")
    moment_list.append("arm_rot_l")
    moment_list.append("elbow_flex_l")
    muscle_analysis.setCoordinates(moment_list)

    # Configure the analysis.
    muscle_analysis.setOn(True)
    muscle_analysis.setStepInterval(1)
    muscle_analysis.setInDegrees(True)
    muscle_analysis.setComputeMoments(True)

    # Create an AnalyzeTool for normal gait.
    analyze_tool = osim.AnalyzeTool()
    analyze_tool.setName("Muscle_Analysis_Deadlift")

    # Set the model file and motion file to analyze.
    analyze_tool.setModelFilename(model_file)
    analyze_tool.setCoordinatesFileName(motion_file)

    # Add the MuscleAnalysis to the AnalyzeTool.
    analyze_tool.updAnalysisSet().cloneAndAppend(muscle_analysis)

    # Directory where results are stored.
    result_dir = os.path.join(parameter['result_dir'], parameter['label'])
    analyze_tool.setResultsDir(result_dir)

    # Configure AnalyzeTool.
    analyze_tool.setReplaceForceSet(False)
    analyze_tool.setSolveForEquilibrium(True)  # 待确定
    analyze_tool.setStartTime(first_time)
    analyze_tool.setFinalTime(last_time)

    external_load_file = parameter['external_load_file']
    print(external_load_file)
    analyze_tool.setExternalLoadsFileName(external_load_file)

    # Print configuration of the AnalyzeTool to an XML file.
    analyze_tool.printToXML(target_xml)

    analyze_tool = osim.AnalyzeTool(target_xml, True)
    result = analyze_tool.run()


if __name__ == '__main__':
    parameter = open_json('opensim_benchpress_chenzui_010.json')
    if 'force_file' not in parameter:
        # point_kinematics(parameter, label='hand_l')
        # point_kinematics(parameter, label='hand_r')
        parameter = generate_force_mot(parameter)
    generate_external_force_xml('force_' + parameter['sport'], parameter['force_file'], parameter['external_load_file'])
    if 'target_inverse_dynamics' in parameter:
        inverse_dynamics(parameter, "ID_" + parameter['sport'], parameter['model_file'], parameter['motion_file'],
                         parameter['external_load_file'], target_file=parameter['target_inverse_dynamics'])
    if 'target_muscle_analysis' in parameter:
        muscle_analysis(parameter, parameter['model_file'], parameter['motion_file'], parameter['external_load_file'],
                        target_xml=parameter['target_muscle_analysis'])
    plt.show()
