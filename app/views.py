#!/usr/bin/python3
# coding:utf-8
from flask import render_template,json,jsonify,request
from app import app
import base64
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import pickle
from PIL import Image,ImageFont, ImageDraw

global_times =0
chinese_word_count = 3755 # 常见汉字个数
checkpoint_dir = './app/train_model/checkpoint/'
code_to_chinese_file = './app/train_model/code_word.pkl' # 文字和对应的编码

test_image_file = './app/image/pred1.png' # 测试图片
pred1_image_file = './app/image/pred1.png'  #预测结果1图片
pred2_image_file = './app/image/pred2.png'  #预测结果2图片
pred3_image_file = './app/image/pred3.png'  #预测结果3图片


# 构建一个三个卷积(3x3) + 三个最大池化层(2x2) +  两个FC层
def build_cnn(top_k):
    # with tf.device('/cpu:0'):
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
    images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1], name='image_batch') # image_size 64x64
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')

    conv_1 = slim.conv2d(images, 64, [3, 3], 1, padding='SAME', scope='conv1')  # image_size 62x62
    max_pool_1 = slim.max_pool2d(conv_1, [2, 2], [2, 2], padding='SAME')      # image_size 31x31
    conv_2 = slim.conv2d(max_pool_1, 128, [3, 3], padding='SAME', scope='conv2')   # image_size 29x29
    max_pool_2 = slim.max_pool2d(conv_2, [2, 2], [2, 2], padding='SAME')      # image_size 15x15
    conv_3 = slim.conv2d(max_pool_2, 256, [3, 3], padding='SAME', scope='conv3')      # image_size 13x13
    max_pool_3 = slim.max_pool2d(conv_3, [2, 2], [2, 2], padding='SAME')      # image_size 7x7

    flatten = slim.flatten(max_pool_3)
    fc1 = slim.fully_connected(slim.dropout(flatten, keep_prob), 1024, activation_fn=tf.nn.tanh, scope='fc1')  # 激活函数tanh
    logits = slim.fully_connected(slim.dropout(fc1, keep_prob),3755, activation_fn=None,scope='fc2') # 无激活函数
    # logits = slim.fully_connected(flatten, FLAGS.charset_size, activation_fn=None, reuse=reuse, scope='fc')
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)) # softmax
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32)) # 计算准确率

    global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
    rate = tf.train.exponential_decay(2e-4, global_step, decay_steps=2000, decay_rate=0.97, staircase=True) #
    train_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss, global_step=global_step) # 自动调节学习率的随机梯度下降算法训练模型
    probabilities = tf.nn.softmax(logits) #

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)
    accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))

    return {'images': images,
            'labels': labels,
            'keep_prob': keep_prob,
            'top_k': top_k,
            'global_step': global_step,
            'train_op': train_op,
            'loss': loss,
            'accuracy': accuracy,
            'accuracy_top_k': accuracy_in_top_k,
            'merged_summary_op': merged_summary_op,
            'predicted_distribution': probabilities,
            'predicted_index_top_k': predicted_index_top_k,
            'predicted_val_top_k': predicted_val_top_k}


def predictPrepare():
    sess = tf.Session()
    graph = build_cnn(top_k=3)
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if ckpt:
        saver.restore(sess, ckpt)
    return graph, sess

def imageprepare(image_path):
    temp_image = Image.open(image_path).convert('L')
    temp_image = temp_image.resize((64, 64), Image.ANTIALIAS)
    temp_image = np.asarray(temp_image) / 255.0
    temp_image = temp_image.reshape([-1, 64, 64, 1])
    return temp_image

def createImage(predword,imagepath):
    im = Image.new("RGB", (64, 64), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    # ttf = '/usr/share/httpd/noindex/css/fonts/Bold/OpenSans-Bold.ttf'
    # fonts = ImageFont.truetype(ttf, 30)
    fonts = ImageFont.truetype("C:\Windows\Fonts\msyh.ttc",36,encoding='utf-8')
    dr.text((15, 10), predword,font=fonts,fill="#000000")
    im.save(imagepath)

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html",title='Home')

@app.route('/chinese_reg',methods=['POST'])
def chinese_reg():
    # 接受前端发来的数据
    data = json.loads(request.form.get('data'))
    imagedata = data["test_image"]
    imagedata = imagedata[22:]
    img = base64.b64decode(imagedata)
    file = open(test_image_file, 'wb')
    file.write(img)
    file.close()
    global global_times

    if (global_times == 0):
        global graph1, sess1
        graph1, sess1 = predictPrepare() #加载模型，准备好预测
        temp_image = imageprepare(test_image_file)
        predict_val, predict_index = sess1.run([graph1['predicted_val_top_k'], graph1['predicted_index_top_k']],
                                              feed_dict={graph1['images']: temp_image, graph1['keep_prob']: 1.0}) # 预测top3的汉字编码以及相应的准确率
        with open(code_to_chinese_file, 'rb') as f2:
            word_dict = pickle.load(f2) # 汉字和编码对照字典
        createImage(word_dict[predict_index[0][0]], pred1_image_file) # 生成准确率top1的汉字图片
        createImage(word_dict[predict_index[0][1]], pred2_image_file)
        createImage(word_dict[predict_index[0][2]], pred3_image_file)
        global_times = 1
    else:
        temp_image = imageprepare(test_image_file)
        predict_val, predict_index = sess1.run([graph1['predicted_val_top_k'], graph1['predicted_index_top_k']],
                                              feed_dict={graph1['images']: temp_image, graph1['keep_prob']: 1.0})
        with open(code_to_chinese_file, 'rb') as f2:
            word_dict = pickle.load(f2)
        createImage(word_dict[predict_index[0][0]], pred1_image_file)
        createImage(word_dict[predict_index[0][1]], pred2_image_file)
        createImage(word_dict[predict_index[0][2]], pred3_image_file)

    # 将识别图片转码传给前端，并带上对应的准确率
    with open(pred1_image_file, 'rb') as fin:
        image1_data = fin.read()
        pred1_image = base64.b64encode(image1_data)
    with open(pred2_image_file, 'rb') as fin:
        image2_data = fin.read()
        pred2_image = base64.b64encode(image2_data)
    with open(pred3_image_file, 'rb') as fin:
        image3_data = fin.read()
        pred3_image = base64.b64encode(image3_data)
    info = dict()
    info['pred1_image'] = "data:image/jpg;base64," + pred1_image.decode()
    info['pred1_accuracy'] = str('{:.2%}'.format(predict_val[0][0]))
    info['pred2_image'] = "data:image/jpg;base64," + pred2_image.decode()
    info['pred2_accuracy'] = str('{:.2%}'.format(predict_val[0][1]))
    info['pred3_image'] = "data:image/jpg;base64," + pred3_image.decode()
    info['pred3_accuracy'] = str('{:.2%}'.format(predict_val[0][2]))
    return jsonify(info)