import h5py
import pandas as pd
import matplotlib.pyplot as plt

# 파일 경로 (현재 디렉토리에 있는 경우)
file_path = "tele_data_0614_act.hdf5"

# HDF5 구조 탐색 함수
def explore_hdf5_structure(g, path="/"):
    structure = {}
    for key in g.keys():
        item = g[key]
        item_path = path + key
        if isinstance(item, h5py.Group):
            structure[item_path + "/"] = "Group"
            structure.update(explore_hdf5_structure(item, item_path + "/"))
        else:
            structure[item_path] = {
                "Type": "Dataset",
                "Shape": item.shape,
                "Dtype": str(item.dtype)
            }
    return structure

# 데이터 읽기
with h5py.File(file_path, 'r') as f:

    cam_high_data = f['/observations/images/cam_high'][:]
    cam_low_data = f['/observations/images/cam_low'][:]
    qpos_data = f['/observations/qpos'][:]
    action_data = f['/action'][:]


# qpos/action을 DataFrame으로 변환
action_columns = [f"action_{i}" for i in range(6)] + ["action_gripper"]
qpos_columns = [f"qpos_{i}" for i in range(6)] + ["gripper_qpos"]

df_action = pd.DataFrame(action_data, columns=action_columns)
df_qpos = pd.DataFrame(qpos_data, columns=qpos_columns)
df_combined = pd.concat([df_qpos, df_action], axis=1)

# DataFrame 저장 또는 출력
df_combined.to_csv("qpos_action_full.csv", index=False)
print("✅ qpos + action 데이터를 'qpos_action_full.csv'로 저장했습니다.")

# 이미지 시각화 (첫 프레임)
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(cam_high_data[0])
axes[0].set_title("cam_high (frame 0)")
axes[0].axis('off')

axes[1].imshow(cam_low_data[0])
axes[1].set_title("cam_low (frame 0)")
axes[1].axis('off')

plt.tight_layout()
plt.show()

# 구조 출력
with h5py.File(file_path, 'r') as f:
    structure = explore_hdf5_structure(f)
    print("\n📁 HDF5 파일 구조:")
    for k, v in structure.items():
        print(f"{k}: {v}")
