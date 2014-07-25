import theano.tensor as T
import numpy as np
import theano

class POS(object):
    def __init__(self):
        self.tag2id={}
        self.window=2
        self.embedding_length=50
        self.word2embedding={}


    def load_embedding(self, embedding_file):
        file = open(embedding_file)
        line_limit=0
        for line in file:
            line_limit+=1
            '''
            if line_limit > 10000:
                break
            '''
            tokens=line.strip().split(' ')
            list=[]
            for i in range(1, len(tokens)):
                list.append(float(tokens[i]))
            self.word2embedding[tokens[0]]=list   
            #del list[:]   is wrong    
        self.embedding_length=len(self.word2embedding['and'])
        file.close()
        print 'embeddings are loaded over!'
    '''
    def address_each_sentence(self, wordlist, taglist):
        gold_labels=np.arange(len(taglist))
        for i in range(len(taglist)):
            if taglist[i] not in self.tag2id.keys():              
                self.tag2id[taglist[i]]=len(self.tag2id) # tag id starts from 0
            gold_labels[i]=self.tag2id[taglist[i]]
            
        value_matrix=np.zeros((len(wordlist), (self.window*2+1)*self.embedding_length))
        for i in range(len(wordlist)):
            neighbor=[]
            for j in range(i-self.window, i+1+self.window):
                if j<0 or j > len(wordlist)-1:
                    neighbor.append('<b>')
                else:
                    neighbor.append(wordlist[j])
            # embedding list
            print neighbor
            value_list=[]
            for token in neighbor:
                word_embedding=[]
                if token not in self.word2embedding.keys():
                    word_embedding = np.random.rand(self.embedding_length)
                else:
                    word_embedding = np.array(self.word2embedding[token])
                value_list.extend(word_embedding)
            value_matrix[i]=value_list
        return value_matrix, gold_labels    
    '''
    def address_each_sentence(self, wordlist, taglist):
        gold_labels=np.arange(len(taglist))
        for i in range(len(taglist)):
            if taglist[i] not in self.tag2id.keys():              
                self.tag2id[taglist[i]]=len(self.tag2id) # tag id starts from 0
            gold_labels[i]=self.tag2id[taglist[i]]
        
        value_matrix=np.zeros((len(wordlist), (self.window*2+1)*self.embedding_length))
        for i in range(len(wordlist)):
            value_list_window=[]
            for j in range(i-self.window, i+1+self.window):
                if j<0 or j > len(wordlist)-1:
                    value_list_window.extend(np.array(self.word2embedding['</s>']))
                else:                   
                    temp=self.word2embedding.get(wordlist[j], -1)
                    if temp ==-1:
                        value_list_window.extend(np.random.rand(self.embedding_length))
                    else:
                        value_list_window.extend(np.array(temp))
                    
                    '''
                    if wordlist[j] not in self.word2embedding:
                        value_list_window.extend(np.random.rand(self.embedding_length))
                    else:
                        value_list_window.extend(np.array(self.word2embedding[wordlist[j]]))
                    '''
            value_matrix[i]=value_list_window
        return value_matrix, gold_labels             

    def read_file(self, file):
        print file
        open_file=open(file)
        representation_list=[]
        gold_label_list=[]
        sentCount=0
        sent_word=[]
        sent_tag=[]
        for line in open_file:
            line = line.strip()
            #print len(line)
            if len(line)==0:
                sentCount+=1              
                '''
                if sentCount>100:
                    break
                '''
                value_matrix, gold_labels=self.address_each_sentence(sent_word, sent_tag)
                if sentCount % 1000 ==0:
                    print 'Sentence: '+str(sentCount)
                #print value_matrix.shape
                representation_list.extend(value_matrix) # a list of matrix
                gold_label_list.extend(gold_labels)
                del sent_word[:]
                del sent_tag[:]
            else:
                tokens=line.split('\t')
                #print len(tokens), line
                sent_word.append(tokens[0].lower()) # convert to lowercase words
                sent_tag.append(tokens[1])
        open_file.close()
        return np.asarray(representation_list), gold_label_list

def load_data_pos(trainFile, devFile, testFile):
    pos=POS()
    #pos.load_embedding('/mounts/data/proj/wenpeng/Tensor/result/word2embedding_calculus.txt')
    pos.load_embedding('/mounts/data/proj/wenpeng/PhraseEmbedding/mikolov-phrases-embeddings.txt')
    #trainFile
    path='/mounts/Users/student/wenpeng/FLORS/datasets/Google Task/'
    train_set_x_raw, train_set_y_raw=pos.read_file(path+trainFile)
    dev_set_x_raw, dev_set_y_raw=pos.read_file(path+devFile)
    test_set_x_raw, test_set_y_raw=pos.read_file(path+testFile)
    
    def shared_dataset(data_x, data_y, borrow=True):
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),  # @UndefinedVariable
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
    
    test_set_x, test_set_y = shared_dataset(train_set_x_raw, train_set_y_raw)
    dev_set_x, dev_set_y = shared_dataset(dev_set_x_raw, dev_set_y_raw)
    train_set_x, train_set_y = shared_dataset(test_set_x_raw, test_set_y_raw)
    

    
    #print train_set_x
    
    rval = [(train_set_x, train_set_y), (dev_set_x, dev_set_y),
            (test_set_x, test_set_y)]
    return rval, len(pos.tag2id), pos.embedding_length, pos.window
    
if __name__ == '__main__':
    load_data_pos('source/ontonotes-wsj-train', 'target/wsj/gweb-wsj-dev', 'target/wsj/gweb-wsj-test')
    
    
    
        