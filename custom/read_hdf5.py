import h5py

# 파일 열기
with h5py.File('/home/chem/act/custom/tele_data_0614_act.hdf5', 'r') as f:
    # 전체 구조 출력
    def print_structure(name, obj):
        print(name)
    f.visititems(print_structure)

    # 올바른 경로로 접근
    print("\n== Action ==")
    print(f['action'].shape)
    print(f['action'][3:])

    print("\n== Observations/images/cam_high ==")
    print(f['observations/images/cam_high'].shape)
    print(f['observations/images/cam_high'][3:])


    print("\n== Observations/images/cam_low ==")
    print(f['observations/images/cam_low'].shape)
    print(f['observations/images/cam_low'][3:])

    print("\n== Observations/qpos ==")
    print(f['observations/qpos'].shape)
    print(f['observations/qpos'][3:])

    # print("\n== Observations/qvel ==")
    # print(f['observations/qvel'][:])
