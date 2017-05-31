import os
from nltk.parse import stanford


def test():
    os.environ['STANFORD_PARSER'] = '/Users/wangfeng/Downloads/stanford-parser-full-2016-10-31/stanford-parser.jar'
    os.environ['STANFORD_MODELS'] = '/Users/wangfeng/Downloads/stanford-parser-full-2016-10-31/stanford-parser-3.7.0-models.jar'
    os.environ['JAVAHOME'] = '/Library/Java/JavaVirtualMachines/jdk1.8.0_111.jdk/Contents/Home'


if __name__ == '__main__':
    test()
    