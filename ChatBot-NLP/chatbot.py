
import tensorflow as tf
import numpy as np
import config
from model_utils import Chatbot
from cornell_data_utils import *
from tqdm import tqdm


--Define get_accuracy helper function to check accuracy of the sequence data

def get_accuracy(target, logits):
    """
    Calculate accuracy
    """
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0,0),(0,max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0,0),(0,max_seq - logits.shape[1])],
            'constant')

    return np.mean(np.equal(target, logits))


--Data cleaning

cleaned_questions, cleaned_answers = clean_data()


--Creating vocab and necessary dictionaries

vocab, word_to_id, id_to_word = create_vocab(cleaned_questions, cleaned_answers)

--Data encoding

encoded_questions = encoder(cleaned_questions, word_to_id)

encoded_answers = encoder(cleaned_answers, word_to_id, True)

--Bucketting data

bucketed_data = bucket_data(encoded_questions, encoded_answers, word_to_id)

--Creating model object, session and defining model saver

model = Chatbot(config.LEARNING_RATE, 
                config.BATCH_SIZE, 
                config.ENCODING_EMBED_SIZE, 
                config.DECODING_EMBED_SIZE, 
                config.RNN_SIZE, 
                config.NUM_LAYERS,
                len(vocab), 
                word_to_id, 
                config.CLIP_RATE) #4=clip_rate

 session = tf.Session()

 session.run(tf.global_variables_initializer())
 saver = tf.train.Saver(max_to_keep=10)


--Entering big buckets, training loop

for i in range(config.EPOCHS):
    epoch_accuracy = []
    epoch_loss = []
    for b in range(len(bucketed_data)):
        bucket = bucketed_data[b]
        questions_bucket = []
        answers_bucket = []
        bucket_accuracy = []
        bucket_loss = []
        for k in range(len(bucket)):
            questions_bucket.append(np.array(bucket[k][0]))
            answers_bucket.append(np.array(bucket[k][1]))
            
        for ii in tqdm(range(len(questions_bucket) //  config.BATCH_SIZE)):
            
            starting_id = ii * config.BATCH_SIZE
            
            X_batch = questions_bucket[starting_id:starting_id+config.BATCH_SIZE]
            y_batch = answers_bucket[starting_id:starting_id+config.BATCH_SIZE]
            
            feed_dict = {model.inputs:X_batch, 
                         model.targets:y_batch, 
                         model.keep_probs:config.KEEP_PROBS, 
                         model.decoder_seq_len:[len(y_batch[0])]*config.BATCH_SIZE,
                         model.encoder_seq_len:[len(X_batch[0])]*config.BATCH_SIZE}
            
            cost, _, preds = session.run([model.loss, model.opt, model.predictions], feed_dict=feed_dict)
            
            epoch_accuracy.append(get_accuracy(np.array(y_batch), np.array(preds)))
            bucket_accuracy.append(get_accuracy(np.array(y_batch), np.array(preds)))
            
            bucket_loss.append(cost)
            epoch_loss.append(cost)
            
        print("Bucket {}:".format(b+1), 
              " | Loss: {}".format(np.mean(bucket_loss)), 
              " | Accuracy: {}".format(np.mean(bucket_accuracy)))
        
    print("EPOCH: {}/{}".format(i, config.EPOCHS), 
          " | Epoch loss: {}".format(np.mean(epoch_loss)), 
          " | Epoch accuracy: {}".format(np.mean(epoch_accuracy)))
    
    saver.save(session, "checkpoint/chatbot_{}.ckpt".format(i))