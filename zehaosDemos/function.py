# 这个函数从数据文件夹中读取所有的图片列表并按训练，验证，测试数据分开
def create_image_lists(testing_percentage):
    result = { }
    # os.walk()的特性获取的是INPUT_DATA的子目录的路径
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        # 获得当前目录下的所有的有效图片文件
        extensions = ['png', 'PNG']
        file_list = []
        # 只是获取路径中最后的文件名
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.'+extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue
        # 通过目录名获取类别的名称
        label_name = dir_name.lower()
        training_images = []
        testing_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            chance = np.random.randint(100)
            if chance < testing_percentage:
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images
        }
    return result

# result记录了训练和测试图片的信息，按照类别分类

# 这个函数通过类别名称，所属数据集和图片编号获取一张图片的地址
# image_lists参数给出了所有图片信息
# image_dir 参数给出了根目录。存放图片数据的根目录和存放图片特征向量
# 的根目录地址不同
# label_name参数制定了类别名称
# index参数给定了需要获取的图片的编号
# category参数制定了需要获取的图片是在训练集还是测试集
def get_image_path(image_lists, image_dir, label_name, index, category):
    label_lists = image_lists[label_name]
    # 根据所属数据集的名称获取集合中的全部图片信息
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    # 获取图片的文件名
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    # 最终的地址为数据根目录的地址加上类别的文件夹加上图片的名称
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


# 这个函数通过类别名称，所属数据集和图片编号获取经过Inception-v3模型处理之后
# 特征向量文件地址
def get_bottleneck_path(image_lists, label_name, index, category):
    return get_image_path(image_lists, CACHE_DIR, label_name, index, category)+'.txt'


# 这个函数使用加载的训练好的Inception-v3模型处理一张图片，得到这个图片的特征向量
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    # 经过卷积神经网络处理的结果是一个四维数组，需要将这个结果压缩成一个特征向量
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


# 将特征向量写入文件，但具体文件的形式需要注意，此处先实验
def write_bottleneck(
        sess, image_lists, label_name, index, category, png_data_tensor, bottleneck_tensor
):
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, category)
    if not os.path.exists(bottleneck_path):
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)
        image_data = gfile.FastGFile(image_path, 'rb').read()
        bottleneck_values = run_bottleneck_on_image(
            sess, image_data, png_data_tensor, bottleneck_tensor)
        # 将计算得到的特征向量存入文件
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)

# 这个函数随机获取一个batch的图片作为训练数据,并产生标签值
def get_random_batch(
        sess, n_classes, image_lists, batch_size, category, png_data_tensor, bottleneck_tensor
):
    inputs = []
    labels = []
    for _ in range(batch_size):
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, image_index, category)
        image_data = gfile.FastGFile(image_path, 'rb').read()
        # 保存读取的图片信息
        inputs.append(image_data)
        # 保存
        labels.append(label_index)
        # write_bottleneck(sess, image_lists, label_name, image_index, category, png_data_tensor, bottleneck_tensor)
        # ground_truth = np.zeros(n_classes, dtype=np.float32)
        # ground_truth[label_index] = 1.0

