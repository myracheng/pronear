import csv
import pandas as pd 
with open('log.txt', 'r') as f:
        l = f.readlines()
    
F1s = []
heur = []
curr_key = 1
curr = {}
iter_c = 0
for ind in range(len(l)):
    line = l[ind]
    if "F1 score achieved is" in line:
        # F1s.append(line[-6:-1])
        curr['F1'] = line[-6:-1]
    if "RNN Heuristic score at Node 1:" in line:
        heur.append(curr)
        iter_c += 1
        curr = {}
        curr_key = 1
        curr[curr_key] = line[-8:-1]
        curr["Iteration"] = iter_c
        curr_key += 1
    elif "RNN Heuristic" in line:
        print(line)
        curr[curr_key] = line[-8:-1]
        curr_key += 1
    if "Node selected" in line:
        curr["node selected"] = line[-3:-1]
        curr["Program"] = l[ind+2][len("INFO:root:"):]
heur.append(curr)
print(F1s)
print(heur)

f = open('short_log.txt',"w")
f.write(str(heur))
f.close()

# h_file = "neurh.csv"
# with open(h_file, "w", newline="\n") as f:
#     writer = csv.writer(f)
#     writer.writerows(F1s)
#     writer.writerows(heur)

df = pd.DataFrame(heur,columns=['Iteration','Initial F1','node selected','Program',].extend(range(curr_key)))
print(df)
df.to_csv('neurh.csv',index=False)
