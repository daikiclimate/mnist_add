import random
import pickle
def main():
    f = open("./data/annotation/train.txt","rb")
    train_list = pickle.load(f)

    f = open("./data/annotation/test.txt","rb")
    test_list = pickle.load(f)

    sep_label = [[] for i in range(10)]
    for i,j in train_list:
        sep_label[i].append(j)
    min_n = []
    for i in sep_label:
        min_n.append(len(i))
        print(len(i),",",end="")
    print("")
    
    pair = []
    for _ in range(1000):
        for i in range(0,10):
            for j in range(0,10):
                
                name1 = sep_label[i][random.randrange(0,min_n[i])]
                name2 = sep_label[j][random.randrange(0,min_n[j])]
                label = [i,j,i+j]
                pair.append([name1, name2, label])
    print(len(pair))
    f = open("data/annotation/train_pair.txt", "wb")
    pickle.dump(pair ,f)

    sep_label = [[] for i in range(10)]
    for i,j in test_list:
        sep_label[i].append(j)
    min_n = []
    for i in sep_label:
        min_n.append(len(i))
    min_num = min(min_n)//100*100
    
    pair = []
    #for k in range(min_num//100):
    for k in range(44):
        for i in range(10):
            for j in range(10):
                name1 = sep_label[i].pop(0)
                name2 = sep_label[j].pop(0)
                label = [i,j,i+j]
                pair.append([name1, name2, label])

    print(len(pair))
    f = open("data/annotation/test_pair.txt", "wb")
    pickle.dump(pair ,f)

    




if __name__ == "__main__":
    main()
