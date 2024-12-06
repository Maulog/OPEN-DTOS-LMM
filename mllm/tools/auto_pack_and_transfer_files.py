import os
import subprocess
import ntpath

local_folder = '/share_ssd1/tianjirui/dtos_output/stage2/exp13/eval_1tgt_method3_nms07_box4' # /checkpoint-50000

is_transfer = True
remote_ip = 'XX.X.XXX.XXX'  # 这里一定记得不要上传
remote_user = 'XXX'
remote_password = 'XXX'
remote_path = r'C:\Users\ai308\Desktop\video\submit'


save_filename_list = ['mevis', 'youtubervos'] # 'davis'   'youtubervos'   'mevis'

def parse_filename(filename):
    path_elements = filename.split('/')
    checkpoint_name = path_elements[-2]
    method_name = path_elements[-1]
    return checkpoint_name, method_name
    

def pack_files(local_folder, checkpoint_name, method_name, dataset_name):
    archive_name = f"{checkpoint_name}_{method_name}_{dataset_name}.zip"
    archive_path = os.path.join('..', archive_name)
    
    if dataset_name == 'mevis':
        os.chdir(os.path.join(os.path.dirname(local_folder), method_name, dataset_name, 'test'))
    elif dataset_name == 'youtubervos':
        os.chdir(os.path.join(os.path.dirname(local_folder), method_name, dataset_name, 'val'))
    zip_command = f"zip -r {archive_path} *"
    
    subprocess.run(zip_command, shell=True, check=True)
    os.chdir(local_folder)

    return archive_name

def transfer_files(archive_name, remote_ip, remote_user, remote_password, remote_path, dataset_name): 
    remote_path = ntpath.join(remote_path, archive_name) # 这里是传到windows上，首先统一分隔符
    remote_path = remote_path.replace('\\', '/') # 因为是在linux上运行，所以替换
    # remote_path = 'C:\\Users\\ai308\\Desktop\\video\\submit\\exp08_eval_tgt1_method3_nms05_ytvos_youtubervos.zip' 
    archive_path = os.path.join(dataset_name, archive_name)
    scp_command = f"sshpass -p {remote_password} scp {archive_path} {remote_user}@{remote_ip}:{remote_path}"
    subprocess.run(scp_command, shell=True, check=True)
    print(f"Transfer {archive_name} to {remote_ip}:{remote_path} successfully!")

def main():
    os.chdir(local_folder)
    checkpoint_name, method_name = parse_filename(local_folder)
    
    for dataset_name in save_filename_list:
        archive_name = pack_files(local_folder, checkpoint_name, method_name, dataset_name)
        if is_transfer:
            transfer_files(archive_name, remote_ip, remote_user, remote_password, remote_path, dataset_name)


if __name__ == "__main__":
    main()