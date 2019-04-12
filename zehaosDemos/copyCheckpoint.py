
import os
import re
import shutil
import time

root = "E:/tmp/cotton-fun-model/"
Dir = "E:/model/test/"
# files = []
# for dirpath, dirnames, filenames in os.walk(root):
#     for filename in filenames:
#         file_path = os.path.join(dirpath, filename)
#         if filename == 'checkpoint' or filename == 'graph.pbtxt':
#             shutil.copy(file_path, targetDir)
#         else:
#             str1 = 'model.ckpt-.*'
#             match_obj = re.match(str1, filename)
#             if match_obj:
#                 files.append(file_path)
# targetFiles = files[-3:]
# for targetFile in targetFiles:
#     print(targetFile)
#     shutil.copy(targetFile, targetDir)

# ------------------------------
while True:
    all_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            all_files.append(filename)

    begin = all_files[-1].index('-')
    # 取出最后一个模型的轮数
    end = all_files[-1].index(".meta")

    name = all_files[-1][begin + 1:end]
    targetDir = Dir + name
    # 如果这个模型不存在
    if not os.path.exists(targetDir):
        os.makedirs(targetDir)
        for file in all_files:
            file_path = os.path.join(root, file)
            if file == 'checkpoint' or file == 'graph.pbtxt':
                shutil.copy(file_path, targetDir)
            else:
                str1 = 'model.ckpt-' + name + ".*"
                match_obj = re.match(str1, file)
                if match_obj:
                    shutil.copy(file_path, targetDir)
    time.sleep(1200)




        #print(image_name)
        # str_cond = 'model.*'
        # part1 = re.compile(str_cond)  # 正则表达式筛选条件
        # if len(part1.findall(filename)):  #
        # match_obj = re.match(str1, file_path)
        # if match_obj:
        #     print(file_path)
        #     shutil.copy(file_path,  targetDir)


