import h5py

def show_hdf5_structure(file_path):
    with h5py.File(file_path, 'r') as f:
        def visitor(name, node):
            if isinstance(node, h5py.Dataset):
                print(f"[DATA] {name}: shape={node.shape}, dtype={node.dtype}")
            elif isinstance(node, h5py.Group):
                print(f"[GROUP] {name}")
        f.visititems(visitor)

file1 = "/home/chem/act/data/rb_transfer_can/episode_3.hdf5"
file2 = "/home/chem/act/data/rb_transfer_can/episode_15.hdf5"
files = [file1, file2]

show_hdf5_structure(file1)

show_hdf5_structure(file2)

for file in files:
    print(f'filepath: {file} \n')
    show_hdf5_structure(file)
    print("\n")