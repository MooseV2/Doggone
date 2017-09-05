# from DoggoneUtils import *
from DoggoneToken import * # TOKEN = <...>
import os
import sys
import tensorflow as tf
import logging
import time
from tendo.singleton import SingleInstance
me = SingleInstance()
from telegram.ext import Updater, MessageHandler, Filters
from telegram import ChatAction
import io


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')



def root_path():
    # Infer the root path from the run file in the project root (e.g. manage.py)
    fn = getattr(sys.modules['__main__'], '__file__')
    root_path = os.path.abspath(os.path.dirname(fn))
    return root_path
print("Root path: [%s]" % root_path())


def photo(bot, update):
    chat_id = update.message.chat_id
    bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    file_id = update.message.photo[-1].file_id
    newFile = bot.getFile(file_id)
    dataFile = io.BytesIO()
    
    # newFile.download(file_id+".jpg")

    print("11111111111")
    print(dataFile.getvalue())
    
    newFile.download(out=dataFile)
    newFile.download(file_id+".jpg")
    print("2222222222")
    print(tf.gfile.FastGFile("/Users/moose/Desktop/WhatIsDogBot/src/"+file_id+".jpg", 'rb').read())
    classification = []
    print("333333333")
    
    print(dataFile.getvalue())
    
    for dog_type, confidence in classify(dataFile.getvalue(), 5):
        classification.append({
            "dog_type": dog_type.title(),
            "confidence": confidence
        })

    message = "I'm %0.2f percent sure that's a %s. If not, my other guesses are: \n" % (classification[0]["confidence"]*100, classification[0]["dog_type"])
    message += "\n".join([" - %s (%0.2f%s)" % (dog["dog_type"], dog["confidence"], "%") for dog in classification[1:]])

    bot.send_message(chat_id=chat_id, text=message)

def classify(imagedata, results=5):
    global graph
    with tf.Session(graph=graph) as sess:
        print(sess)
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': imagedata})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1][:results]

        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            # print('%s (score = %.5f)' % (human_string, score))

        return [(label_lines[node_id], predictions[0][node_id]) for node_id in top_k]


if __name__ == "__main__":
    
    # Files
    dog_graph = os.path.join(root_path(), "nn_dog_graph.pb")
    dog_labels = os.path.join(root_path(), "nn_dog_labels.txt")

    # Load imagenet classifier
    with tf.gfile.FastGFile(dog_graph, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        del(graph_def.node[1].attr["dct_method"])
        _ = tf.import_graph_def(graph_def, name='')
        # Load labels
        graph = tf.get_default_graph()

    label_lines = [line.rstrip() for line in tf.gfile.GFile(dog_labels)]

    updater = Updater(token=TOKEN)
    dispatcher = updater.dispatcher
    dispatcher.add_handler(MessageHandler(Filters.photo, photo))
    updater.start_polling()

