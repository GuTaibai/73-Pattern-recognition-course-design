'''
from PIL import Image

def resize_and_merge_images(image_paths, output_path):
    # 打开所有图片并获取它们的最大尺寸
    images = [Image.open(path) for path in image_paths]
    max_width = max(image.width for image in images)
    max_height = max(image.height for image in images)

    # 计算合并后图片的尺寸
    total_width = max_width * 2
    total_height = max_height * 2

    # 创建一张新的空白图片用于合并
    merged_image = Image.new('RGB', (total_width, total_height))

    # 将图片缩放并粘贴到新图片上
    resized_images = [image.resize((max_width, max_height), Image.LANCZOS) for image in images]
    merged_image.paste(resized_images[0], (0, 0))
    merged_image.paste(resized_images[1], (max_width, 0))
    merged_image.paste(resized_images[2], (0, max_height))
    merged_image.paste(resized_images[3], (max_width, max_height))

    # 保存合并后的图片
    merged_image.save(output_path)

# 输入的四张图片路径
image_paths = ['7.jpg', '8.jpg', '9.jpg', '10.jpg']

# 输出图片路径
output_path = 'merged_image3.jpg'

# 调用函数进行处理
resize_and_merge_images(image_paths, output_path)

'''
from PIL import Image

def resize_and_merge_images(image_paths, output_path):
    # 打开两张图片并获取它们的最大尺寸
    images = [Image.open(path) for path in image_paths]
    max_width = max(image.width for image in images)
    max_height = max(image.height for image in images)

    # 计算合并后图片的尺寸
    total_width = max_width * 2
    total_height = max_height

    # 创建一张新的空白图片用于合并
    merged_image = Image.new('RGB', (total_width, total_height))

    # 将图片缩放并粘贴到新图片上
    resized_images = [image.resize((max_width, max_height), Image.LANCZOS) for image in images]
    merged_image.paste(resized_images[0], (0, 0))
    merged_image.paste(resized_images[1], (max_width, 0))

    # 保存合并后的图片
    merged_image.save(output_path)

# 输入的两张图片路径
image_paths = ['11.jpg', '12.jpg']

# 输出图片路径
output_path = 'merged_image4.jpg'

# 调用函数进行处理
resize_and_merge_images(image_paths, output_path)

