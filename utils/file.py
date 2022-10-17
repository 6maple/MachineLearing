# coding = utf-8
import os

def un_gz(file_path, output_dir=None):
    """解压.gz文件
    
    :param file_path:            文件路径
    :param output_dir:           输出目录
    """
    import gzip
    # 获取文件名和目录
    directory, file_name = os.path.split(file_path)
    output_name, _ = os.path.splitext(file_name)
    # 设置输出目录
    if output_dir is None:
        output_dir = directory
    # 设置输出路径
    output_path = os.path.join(output_dir, output_name)
    # 使用gzip对象来完成解压
    with gzip.GzipFile(file_path) as gzip_file:
        with open(output_path, 'wb+') as fp:
            fp.write(gzip_file.read())
    
def un_tar(file_path, output_dir=None):
    """解压.tar文件
    
    :param file_path:            文件路径
    :param output_dir:           输出目录
    """
    import tarfile
    # 获取文件名和目录
    directory, file_name = os.path.split(file_path)
    output_name, _ = os.path.splitext(file_name)
    # 设置输出目录
    if output_dir is None:
        output_dir = directory
    # 设置输出路径
    output_path = os.path.join(output_dir, output_name)
    # 创建输出目录
    os.mkdir(output_path)
    # 使用tarfile解压
    with tarfile.open(file_path) as tar_fp:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar_fp, output_path)
            
def un_zip(file_path, output_dir=None):
    """解压.tar文件
    
    :param file_path:            文件路径
    :param output_dir:           输出目录
    """
    import zipfile
    # 获取文件名和目录
    directory, file_name = os.path.split(file_path)
    output_name, _ = os.path.splitext(file_name)
    # 设置输出目录
    if output_dir is None:
        output_dir = directory
    # 设置输出路径
    output_path = os.path.join(output_dir, output_name)
    # 创建输出目录
    os.mkdir(output_path)
    # 使用zipfile解压
    with zipfile.ZipFile(file_path) as zip_fp:
        zip_fp.extractall(output_path)
            
def un_rar(file_path, output_dir=None):
    """解压.rar文件
    
    :param file_path:            文件路径
    :param output_dir:           输出目录
    """
    import rarfile
    # 获取文件名和目录
    directory, file_name = os.path.split(file_path)
    output_name, _ = os.path.splitext(file_name)
    # 设置输出目录
    if output_dir is None:
        output_dir = directory
    # 设置输出路径
    output_path = os.path.join(output_dir, output_name)
    # 创建输出目录
    os.mkdir(output_path)
    # 使用rarfile解压
    with rarfile.RarFile(file_path) as rar_fp:
        rar_fp.extractall(output_path)
        