import jieba
import GrobalParament
import re
import pandas as pd


def delete_r_n(line):
    return line.replace('\r', '').replace('\n', '').strip()


def get_stop_words(stop_words_dir):
    """
    读取停用词
    :param stop_words_dir:
    :return:
    """
    stop_words = []
    with open(stop_words_dir, 'r', encoding=GrobalParament.encoding) as f:
        for line in f.readlines():
            line = delete_r_n(line)
            if line not in stop_words:
                stop_words.append(line)
    return stop_words

def jieba_cut(content, stop_words):
    """
    jieba精准分词
    :param content:
    :param stop_words:
    :return:
    """
    word_list = []
    if content != '' and content is not None:
        seg_list = jieba.cut(content)
        for word in seg_list:
            if word not in stop_words:
                word_list.append(word)
    return word_list


def jieba_cut_for_search(content, stop_words):
    """
    jieba 搜索引擎分词
    :param content:
    :param stop_words:
    :return:
    """
    word_list = []
    if content != '' and content is not None:
        seg_list = jieba.cut_for_search(content)
        for word in seg_list:
            if word not in stop_words:
                word_list.append(word)
    return word_list


def find_sentences(text):
    """
    找到完整句子
    :param text:
    :return:
    """
    sentences = re.split(r'(\.|\!|\?|。|！|？|\.{6})', text)
    return sentences


def clear_word_from_vocab(word_list, vocab):
    """
    清除不在词库中的词
    :param word_list:
    :param vocab:
    :return:
    """
    new_word_list = []
    for word in word_list:
        if word in vocab:
            new_word_list.append(word)
    return new_word_list


def preprocessing_text_txt(text_dir, after_preprocess_dir, stop_dir):
    """
    文本预处理
    :param text_dir:
    :param after_preprocess_dir:
    :param stop_dir:
    :return:
    """

    stop_words = get_stop_words(stop_dir)
    sentences = []

    with open(text_dir, 'r', encoding=GrobalParament.encoding) as f:
        with open(after_preprocess_dir, 'w', encoding=GrobalParament.encoding) as f1:
            for i, line in enumerate(f.readlines()):
                line = line.split('\t')[0]
                line = delete_r_n(line)
                word_list = jieba_cut(line, stop_words)
                if len(word_list) > 0:
                    sentences.append(word_list)
                    word = ' '.join(word_list)
                    f1.write(str(i) + ',' + word + '\n')
            print('写入完成')

    return sentences

def preprocessing_text_pd(text_dir, after_preprocess_dir, stop_dir):
    """
    pandas处理数据
    :param text_dir:
    :param after_preprocess_dir:
    :param stop_dir:
    :return:
    """

    stop_words = get_stop_words(stop_dir)
    sentences = []

    df = pd.read_csv(text_dir)
    for index, row in df.iterrows():
        title = delete_r_n(row['title'])
        word_list = jieba_cut(title, stop_words)
        df.loc[index, 'title'] = ' '.join(word_list)
        sentences.append(word_list)
        
    df.to_csv(after_preprocess_dir, encoding=GrobalParament.encoding, index=False)

    return sentences

if __name__ == "__main__":
    stop_words = get_stop_words(GrobalParament.stop_word_dir)
    sentences = preprocessing_text_txt(GrobalParament.train_set_dir, GrobalParament.train_after_process_text_dir, GrobalParament.stop_word_dir)
    print(sentences[:5])