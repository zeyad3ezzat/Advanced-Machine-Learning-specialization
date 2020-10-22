import os
from sklearn.metrics.pairwise import pairwise_distances_argmin

from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from utils import *


class ThreadRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = load_embeddings(paths['WORD_EMBEDDINGS'])
        self.thread_embeddings_folder = paths['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + ".pkl")
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """ Returns id of the most similar thread for the question.
            The search is performed across the threads with a given tag.
        """
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)

        # HINT: you have already implemented a similar routine in the 3rd assignment.

        question_vec = question_to_vec(question, embeddings, dim)[np.newaxis, :]
#### YOUR CODE HERE ####

        best_thread = pairwise_distances_argmin(question_vec, thread_embeddings, metric='cosine')[0]
 #### YOUR CODE HERE ####
        
        return thread_ids[best_thread]


class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")

        # Intent recognition:
        self.intent_recognizer = unpickle_file(paths['INTENT_RECOGNIZER'])
        self.tfidf_vectorizer = unpickle_file(paths['TFIDF_VECTORIZER'])

        self.ANSWER_TEMPLATE = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(paths['TAG_CLASSIFIER'])
        self.thread_ranker = ThreadRanker(paths)
        self.__init_chitchat_bot()

    def __init_chitchat_bot(self):
      
        """Initializes self.chitchat_bot with some conversational model."""

        # Hint: you might want to create and train chatterbot.ChatBot here.
        # Create an instance of the ChatBot class.
        # Create a trainer (chatterbot.trainers.ChatterBotCorpusTrainer) for the ChatBot.
        # Train the ChatBot with "chatterbot.corpus.english" param.
        
        ########################
        #### YOUR CODE HERE ####
        ########################
        # chatbot = 

# Create a new trainer for the chatbot
        # trainer = ChatterBotCorpusTrainer(chatbot)

        # Train the chatbot based on the english corpus
        # trainer.train("chatterbot.corpus.english")

        # Get a response to an input statement
        # chatbot.get_response("Hello, how are you today?")
        self.chitchat_bot=ChatBot('Ron Obvious')

        # remove this when you're done
        # raise NotImplementedError(
        #     "Open dialogue_manager.py and fill with your code. In case of Google Colab, download"
        #     "(https://github.com/hse-aml/natural-language-processing/blob/master/project/dialogue_manager.py), "
        #     "edit locally and upload using '> arrow on the left edge' -> Files -> UPLOAD")

    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""

        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the question.
        
        prepared_question = text_prepare(question)#### YOUR CODE HERE ####
        features = self.tfidf_vectorizer.transform([prepared_question])#### YOUR CODE HERE ####
        intent = self.intent_recognizer.predict(features)#### YOUR CODE HERE ####
        # Chit-chat part:   
        if intent == 'dialogue':
            # Pass question to chitchat_bot to generate a response.       
            response = self.chitchat_bot.get_response(prepared_question)
            return response
        
        # Goal-oriented part:
        else:        
            # Pass features to tag_classifier to get predictions.
            tag = self.tag_classifier.predict(features)#### YOUR CODE HERE ####
            
            # Pass prepared_question to thread_ranker to get predictions.
            thread_id = self.thread_ranker.get_best_thread( prepared_question, tag)#### YOUR CODE HERE ####
            
            return self.ANSWER_TEMPLATE % (tag, thread_id)
