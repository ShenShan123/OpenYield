import os
import shutil

def delete_folder_content(folder_path):
    # 检查文件夹是否存在
    if os.path.exists(folder_path):
        # 遍历文件夹中的所有文件和子文件夹
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    # 如果是文件或链接，直接删除
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    # 如果是文件夹，递归删除
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

# 调用示例
folder_path = '/home/lixy/sim'
delete_folder_content(folder_path)
    