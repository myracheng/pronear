f = open('../../../Downloads/waveform.txt','r')
# print(f[0])
f2 = open('waveform_mod.txt','w')

out = f.readlines() 

def listToString(s):  
    # initialize an empty string 
    str1 = ""  
    
    # traverse in the string   
    for ele in s:  
        str1 += ele   
    
    # return string   
    return str1  

for line in out[:5]:
    if len(line.split('e-02')) == 2:
        a = line.split('e-02')
        a[0] = float(a[0])*0.01
        print(a)
        a[0] = str(a[0])
        f2.write(listToString(a))
    elif len(line.split('e-03')) == 2:
        a = line.split('e-03')
        a[0] = float(a[0])*0.001
        a[0] = str(a[0])
        # print(a)
        f2.write(listToString(a))