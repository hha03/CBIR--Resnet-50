from flask import Flask, request, jsonify, send_from_directory, render_template, flash, redirect, url_for,  session
from werkzeug.utils import secure_filename
import os
import numpy as np
import pickle
from keras._tf_keras.keras.applications.resnet50 import preprocess_input
from keras._tf_keras.keras.applications.resnet50 import ResNet50
from keras._tf_keras.keras.preprocessing import image
from PIL import Image  
import random
import string
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import mysql.connector
import shutil


app = Flask(__name__)

BASE_URL = "http://127.0.0.1:5000"

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD = os.path.join(APP_ROOT, 'uploads')
DATA = os.path.join(APP_ROOT, 'data')
ADD_NEW = os.path.join(APP_ROOT, 'add_new')
app.config['UPLOAD'] = UPLOAD
app.config['DATA'] = DATA
app.config['ADD_NEW'] = ADD_NEW


def ensure_folder_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

ensure_folder_exists(app.config['UPLOAD'])
ensure_folder_exists(app.config['ADD_NEW'])

#connect database
def get_db_connection():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="internship"
    )
    return conn

def get_feature_vectors():
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """SELECT id, feature FROM foods"""
    cursor.execute(query)
    rows = cursor.fetchall()
    
    ids = []
    feature_vectors = []
    for row in rows:
        id = row[0]
        ids.append(id)
        feature_blob = row[1]
        feature_vector = pickle.loads(feature_blob)
        feature_vectors.append(np.array(feature_vector))
    
    conn.close()
    return ids, feature_vectors

ids, features = get_feature_vectors()

# Kích thước ảnh đầu vào 
img_width, img_height = 224,224

# Khởi tạo mô hình ResNet50 và bỏ đi lớp fully connected (top layer)
model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Chuyển đổi thành mảng numpy
X_train = np.array(features)

# Trích xuất đặc trưng từ mô hình ResNet50
def extract_features(image_path):
    img = image.load_img(image_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array, verbose=0)
    return features.flatten()

def get_food_info(id):
    conn = get_db_connection()
    cursor = conn.cursor()
    query_select = """
    SELECT food_name, description, imgPath FROM foods
    WHERE id = %s
    """
    cursor.execute(query_select, (id,))
    food_data = cursor.fetchone()
    conn.close()
    
    if food_data:
        return food_data[0], food_data[1], food_data[2]
    else:
        return "", "", ""

def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD'], filename)

@app.route('/data/<folder>/<filename>')
def data_image(folder,filename):
    return send_from_directory(app.config['DATA'], folder + '/' +filename)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/search-similar', methods=['POST'])
def search_similar():
    file = request.files.get('image')
    if file:
        filename = secure_filename(file.filename)
        file_name_random = get_random_string(12) + filename
        filepath = os.path.join(app.config['UPLOAD'], file_name_random)
        file.save(filepath)

        # Khi có một ảnh mới đầu vào
        new_image_path = filepath

        # Trích xuất đặc trưng của ảnh mới
        new_image_feature = extract_features(new_image_path)

        x_test = [new_image_feature]

        # Tính toán cosine similarity giữa vector đặc trưng của hình ảnh mới và tất cả các hình ảnh trong X_train
        cos_similarities = cosine_similarity(x_test, X_train)

        # Sắp xếp các món ăn dựa trên cosine similarity
        sorted_indices = np.argsort(cos_similarities[0])[::-1]

        # Lấy danh sách các món ăn tương tự 
        num_similar_items = 40
        similar_items = sorted_indices[:num_similar_items]

        list_image_urls = []
        list_food_names = []
        descriptions = []
        list_images = []

        # Duyệt qua từng đường dẫn ảnh trong danh sách similar_items
        for i in similar_items:
            food_id = ids[i]
            food_name, description, img_path = get_food_info(food_id)

            list_food_names.append(food_name.title())
            descriptions.append(description if description else "")
            list_images.append(img_path)

            if 'data' in img_path:
                imageUrl = 'http://' + BASE_URL + '/data' + img_path.split('data')[1].replace("\\", "/")
                list_image_urls.append(imageUrl)
            else:
                imageUrl = 'http://' + BASE_URL + '/data' + img_path.replace("\\", "/")
                

        return jsonify({
            'list_images': list_images,
            'list_image_urls': list_image_urls,
            'list_food_names': list_food_names,
            'descriptions': descriptions
        })
    else:
        return jsonify({'message': 'No file uploaded!'})

#Add new image 

# def get_food_id(food_name):
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     try:
#         # Sử dụng COLLATE để so sánh không phân biệt chữ hoa/thường
#         query_select = "SELECT id FROM food WHERE name LIKE %s COLLATE utf8_general_ci"
#         cursor.execute(query_select, (food_name,))
#         result = cursor.fetchone()
#         cursor.close()
#         conn.close()
#         if result:
#             return result[0]
#         else:
#             return None
#     except mysql.connector.Error as err:
#         print(f"Lỗi: {err}")
#         return None

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    # Kiểm tra xem file có định dạng được cho phép hay không
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/add-image', methods=['POST'])
def add_image():
    file = request.files['imageFile']
    food_name = request.form.get('fname')
    description = request.form.get('description')

    if not allowed_file(file.filename):
        return jsonify({'error': 'File format not allowed'}), 400

    # Kiểm tra xem món ăn đã tồn tại trong cơ sở dữ liệu chưa
    # food_id = get_food_id(food_name)

    conn = get_db_connection()
    cursor = conn.cursor()

    if file:
        # Tạo tên file ngẫu nhiên
        filename = secure_filename(file.filename)
        file_name_random = get_random_string(12) + "_" + filename
        
        # Lưu file tạm thời trong thư mục `ADD_NEW`
        file_path = os.path.join(app.config['ADD_NEW'], file_name_random)
        file.save(file_path)

        # Bước 2: Trích xuất đặc trưng từ ảnh đã lưu trong thư mục `add_new`
        feature_vector = np.array(extract_features(file_path))

        # Bước 3: Di chuyển file từ `add_new` sang `data/images`
        final_path = os.path.join('data', 'images', file_name_random)  # Đường dẫn tương đối
        final_absolute_path = os.path.join(app.config['DATA'], 'images', file_name_random)  # Đường dẫn tuyệt đối

        shutil.move(file_path, final_absolute_path)

        # Chuyển đổi đặc trưng thành nhị phân để lưu vào cơ sở dữ liệu
        feature_blob = pickle.dumps(feature_vector)

        # Lưu thông tin vào cơ sở dữ liệu (tên món ăn, mô tả, đường dẫn ảnh và đặc trưng)
        query_insert_image = """
        INSERT INTO foods (food_name, description, imgPath, feature) 
        VALUES (%s, %s, %s, %s)
        """
        cursor.execute(query_insert_image, (food_name, description, final_path, feature_blob))
        conn.commit()

        # Đóng kết nối và con trỏ
        cursor.close()
        conn.close()

        return jsonify({'success': 'File uploaded successfully', 'filename': file_name_random}), 200
    else:
        return jsonify({'error': 'No file uploaded'}), 400
    
# load random imgs

def get_food():
    connection = get_db_connection()
    cursor = connection.cursor()

    cursor.execute("SELECT food_name, imgPath, description FROM foods")
    foods = cursor.fetchall()  # Lấy tất cả kết quả của query

    # Đóng kết nối sau khi truy vấn xong
    cursor.close()
    connection.close()
    return foods

@app.route('/random-food-images', methods=['GET'])
def get_random_food_images():
    all_foods = get_food()
    random_foods = random.sample(all_foods, 60) if len(all_foods) > 60 else all_foods

    # Tạo danh sách các thông tin món ăn
    foodName = [food[0].title() for food in random_foods]
    list_imagePath = [food[1] for food in random_foods]
    foodDescription = [food[2] if food[2] is not None else "" for food in random_foods]

    return jsonify({
        'foodName': foodName,
        'list_imagePath': list_imagePath,
        'foodDescription': foodDescription
    })


if __name__ == "__main__":
    app.run(debug=True)