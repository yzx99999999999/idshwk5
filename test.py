from sklearn.ensemble import RandomForestClassifier
import numpy as np
import math

domainlist = []
class Domain:
    def __init__(self,_name,_label,_length,_numbers,_entropy):
        self.name = _name
        self.label = _label
        self.length=_length
        self.numbers=_numbers
        self.entropy=_entropy

    def returnData(self):
        return [self.length, self.numbers, self.entropy]

    def returnLabel(self):
        if self.label == "dga":
            return 1
        else:
            return 0

def initData(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line =="":
                continue
            tokens = line.split(",")
            name = tokens[0]
            label = tokens[1]
            length= len(name)
            num=0
            entro=0 
            result={}
            frequency={}
            for i in name:
                result[i]=name.count(i)
                frequency[i]=float(result[i])/length
                entro-=1*(frequency[i])*math.log(frequency[i],2)
                if i.isdigit():
                        num +=1
            numbers= num  
            entropy = entro
            domainlist.append(Domain(name,label,length,numbers,entropy))

def main():
    initData("train.txt")
    featureMatrix = []
    labelList = []
    for item in domainlist:
        featureMatrix.append(item.returnData())
        labelList.append(item.returnLabel())

    clf = RandomForestClassifier(random_state=0)
    clf.fit(featureMatrix,labelList)

    file1=open("test.txt")
    file2=open("result.txt","w")
    for line in file1:
        line = line.strip()
        if line.startswith("#") or line =="":
            continue
        tokens_test = line.split(",")
        name_test = tokens_test[0]
        length_test = len(name_test)
        num_test=0
        entro_test=0     
        result_test={} 
        frequency_test={}
        for i in name_test:
            result_test[i]=name_test.count(i)
            frequency_test[i]=float(result_test[i])/length_test
            entro_test-=1*(frequency_test[i])*math.log(frequency_test[i],2)
            if i.isdigit():
                num_test +=1
        numbers_test= num_test  
        entropy_test = entro_test
        if clf.predict([[length_test,numbers_test,entropy_test]])==0:
            label_test1 = "notdga"
            content=line+","+label_test1
            file2.write(content)
            file2.write("\n")
        if clf.predict([[length_test,numbers_test,entropy_test]])==1:
            label_test2 = "dga"
            content=line+","+label_test2
            file2.write(content)
            file2.write("\n")
    file1.close
    file2.close

if __name__ == '__main__':
    main()
