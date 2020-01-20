
import pickle
def main():
    f = open("./train.txt","rb")
    train_list = pickle.load(f)

    f = open("./test.txt","rb")
    test_list = pickle.load(f)

    sep_label = [[] for i in range(10)]
    for i,j in train_list:
        sep_label[i].append(j)
    min_n = []
    for i in sep_label:
        min_n.append(len(i))
    min_num = min(min_n)//100*100
    
    pair = []
    for k in range(min_num//100):
        for i in range(10):
            for j in range(10):
                name1 = sep_label[i].pop(0)
                name2 = sep_label[j].pop(0)
                label = [i,j,i+j]
                pair.append([name1, name2, label])
    f = open("train_pair.txt", "wb")
    pickle.dump(pair ,f)

    sep_label = [[] for i in range(10)]
    for i,j in test_list:
        sep_label[i].append(j)
    min_n = []
    for i in sep_label:
        min_n.append(len(i))
    min_num = min(min_n)//100*100
    
    pair = []
    for k in range(min_num//100):
        for i in range(10):
            for j in range(10):
                name1 = sep_label[i].pop(0)
                name2 = sep_label[j].pop(0)
                label = [i,j,i+j]
                pair.append([name1, name2, label])


    f = open("test_pair.txt", "wb")
    pickle.dump(pair ,f)

    




if __name__ == "__main__":
    main()
