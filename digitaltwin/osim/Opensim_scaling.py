import opensim as osim
import numpy as np

def vec3_to_np(v):
    """将OpenSim Vec3转换为numpy数组"""
    return np.array([v.get(i) for i in range(3)], dtype=float)

def compute_limb_lengths(model, state):
    """计算肢体长度"""
    # 获取关节位置
    shoulder_joint = model.getJointSet().get("acromial_r")
    elbow_joint = model.getJointSet().get("elbow_r")
    wrist_joint = model.getJointSet().get("radius_hand_r")
    
    shoulder_pos = vec3_to_np(shoulder_joint.getParentFrame().getTransformInGround(state).p())
    elbow_pos = vec3_to_np(elbow_joint.getParentFrame().getTransformInGround(state).p())
    wrist_pos = vec3_to_np(wrist_joint.getParentFrame().getTransformInGround(state).p())
    
    # 计算长度
    uarm_length = np.linalg.norm(elbow_pos - shoulder_pos)
    farm_length = np.linalg.norm(wrist_pos - elbow_pos)
    
    return uarm_length, farm_length, (shoulder_pos, elbow_pos, wrist_pos)

def scale_opensim_model(model_path, target_uarm_length, target_farm_length, output_path):
    # 1. 加载模型
    print("1. 加载标准模型...")
    model = osim.Model(model_path)
    state = model.initSystem()
    
    # 2. 计算当前长度
    print("2. 计算对应尺寸...")
    current_uarm, current_farm, joint_positions = compute_limb_lengths(model, state)
    
    # 3. 计算缩放因子
    print("3. 计算缩放因子...")
    scale_uarm = target_uarm_length / current_uarm
    scale_farm = target_farm_length / current_farm
    
    print(f"上臂缩放因子: {scale_uarm:.6f}")
    print(f"前臂缩放因子: {scale_farm:.6f}")
    
    # 4. 执行缩放
    print("4. 执行模型缩放...")
    
    # 创建ScaleSet进行精确缩放
    scale_set = osim.ScaleSet()
    
    # 定义体段和对应的缩放因子
    segment_scales = {
        # 上臂体段
        "humerus_r": scale_uarm,
        "humerus_l": scale_uarm,
        # 前臂体段  
        "radius_r": scale_farm,
        "ulna_r": scale_farm, 
        "hand_r": scale_farm,
        "radius_l": scale_farm,
        "ulna_l": scale_farm,
        "hand_l": scale_farm
    }
    
    # 添加缩放规则到ScaleSet
    for segment_name, factor in segment_scales.items():
        segment_scale = osim.Scale()
        segment_scale.setSegmentName(segment_name)
        segment_scale.setScaleFactors(osim.Vec3(factor, factor, factor))
        scale_set.adoptAndAppend(segment_scale)
    
    # 应用缩放
    try:
        model.scale(state, scale_set, False)
        print("ScaleSet缩放成功!")
    except Exception as e:
        print("ScaleSet缩放失败")
        return None
    
    # 5. 调整质量属性
    print("5. 调整质量属性...")
    body_set = model.getBodySet()
    
    for segment_name, factor in segment_scales.items():
        if body_set.contains(segment_name):
            body = body_set.get(segment_name)
            try:
                # 按体积比例调整质量 (factor^3)
                original_mass = body.getMass()
                new_mass = original_mass * (factor ** 3)
                body.setMass(new_mass)
                print(f"{segment_name}: 质量 {original_mass:.3f} -> {new_mass:.3f} kg")
            except Exception as e:
                print(f"{segment_name} 质量调整失败: {e}")
    
    # 6. 重新初始化并保存
    print("6. 保存...")
    model.finalizeConnections()
    state = model.initSystem()
    
    # 保存模型
    model.printToXML(output_path)
    print(f"缩放后的模型已保存: {output_path}")
    
    # 7. 验证结果
    print("7. 验证缩放结果...")
    final_uarm, final_farm, _ = compute_limb_lengths(model, state)
    
    uarm_error = abs(final_uarm - target_uarm_length) / target_uarm_length * 100
    farm_error = abs(final_farm - target_farm_length) / target_farm_length * 100
    
    print(f"最终上臂长度: {final_uarm:.3f} m (误差: {uarm_error:.2f}%)")
    print(f"最终前臂长度: {final_farm:.3f} m (误差: {farm_error:.2f}%)")
    
    # 判断缩放是否成功
    if uarm_error < 1.0 and farm_error < 1.0:
        print("缩放成功! 误差在可接受范围内 (<1%)")
        success = True
    else:
        print("缩放精度不足")
        success = False
        
    return model if success else None


if __name__ == "__main__":
    # 配置参数
    model_file_path = "E:/VSCode/Projects/OpenSimRealtimeDisplay/Model/whole body model_DisplayOnly"
    input_model = model_file_path + "/model.osim"
    output_model = model_file_path + "/personalized_model_scaled.osim"
    
    # 目标尺寸 (单位: 米)
    target_upper_arm_length = 0.26878  # 上臂长度
    target_forearm_length = 0.231647  # 前臂长度
    
    # 执行缩放
    scaled_model = scale_opensim_model(
        model_path=input_model,
        target_uarm_length=target_upper_arm_length, 
        target_farm_length=target_forearm_length,
        output_path=output_model
    )
    
    if scaled_model:
        print("\n 模型缩放完成!")
        print(f"可以使用缩放后的模型: {output_model}")
    else:
        print("\n 模型缩放失败")