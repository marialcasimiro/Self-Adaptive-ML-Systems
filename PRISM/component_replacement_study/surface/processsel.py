####################################################
# PRISM surface merger v0.1
# Javier Camara - ISR - Carnegie Mellon University
#
####################################################

import sys
import math
from os.path import basename

print("####################################################")
print("# PRISM surface merger v0.1")
print("# Javier Camara - ISR - Carnegie Mellon University")
print("#####################################################")

colors=['black','white','red']

if len(sys.argv)<4:
    print("Missing arguments")
    print("Use: python merge.py input_file1 input_file2 output_file template_file")
    sys.exit()

src1_file=sys.argv[1]   # repair/retrain
src2_file=sys.argv[2]   # nop
src3_file=sys.argv[3]   # nop
tgt_file=sys.argv[4]
template_file=sys.argv[5]

print(tgt_file)
print(template_file)

DEBUG=1
step=21
max_duration = 100
pointsa=[]
pointsb=[]
pointsc=[]


def str_point2d(point):
    return "("+str(point[0])+","+str(point[1])+")"

def load_points (filename):
    listname=[]
    chunks=[]
    f=open(filename,"r")
    for line in f.readlines()[1:]:
        chunks=line.strip().split("\t")
        point=[]
        if int(chunks[0]) <= max_duration:
            if DEBUG:
                print(str(chunks))
            point.append(float(chunks[0]))
            point.append(float(chunks[1]))
            point.append(round(float(chunks[2]), 3))
            if DEBUG:
                print(str(point))
            listname.append(point)        
    return listname
        
pointsa=load_points(src1_file)
# print(pointsa)
pointsb=load_points(src2_file)
pointsc=load_points(src3_file)

text=""
count_a=0
count_b=0
count_c=0

# text=text+"\n\\addplot+[only marks,mark=square*, mark options={fill="+colors[0]+",draw=black}] coordinates{"

# for i in range(len(pointsa)):
#     if pointsa[i][2]>pointsb[i][2]:
#         count_a=count_a+1
#         text=text+str_point2d(pointsa[i])
            
# text=text+"};\n\n"

# text=text+"\n\\addplot+[only marks,mark=square*, mark options={fill="+colors[1]+",draw=black}] coordinates{"

# for i in range(len(pointsa)):
#     if pointsa[i][2]<=pointsb[i][2]:
#         count_b=count_b+1
#         text=text+str_point2d(pointsa[i])
            
# text=text+"};\n\n"

# TACTIC 1
text=text+"\n\\addplot+[only marks,mark=square*, mark options={fill="+colors[0]+",draw=black}] coordinates{"

for i in range(len(pointsa)):
    if pointsa[i][2]<=pointsc[i][2] and pointsa[i][2]<=pointsb[i][2]:
        count_a=count_a+1
        text=text+str_point2d(pointsa[i])

# if count_a == 0:
#     text=text+"(0.0,1.0)"
            
text=text+"};\n\n"

# TACTIC 2
text=text+"\n\\addplot+[only marks,mark=square*, mark options={fill="+colors[1]+",draw=black}] coordinates{"

for i in range(len(pointsa)):
    if  pointsb[i][2]<pointsc[i][2] and pointsb[i][2]<pointsa[i][2]:
        count_b=count_b+1
        text=text+str_point2d(pointsb[i])

if count_b == 0:
    text=text+"(100.0,100.0)"

text=text+"};\n\n"

# TACTIC 3
text=text+"\n\\addplot+[only marks,mark=square*, mark options={fill="+colors[2]+",draw=black}] coordinates{"


for i in range(len(pointsa)):
    if pointsc[i][2]<pointsa[i][2] and pointsc[i][2]<pointsb[i][2]:
        count_c=count_c+1
        text=text+str_point2d(pointsc[i])

# if count_c == 0:
#     text=text+"(0.0,1.0)"
            
text=text+"};\n\n"




# text=text+"\n\\addplot+[only marks,mark=square*, mark options={fill="+colors[2]+",draw=black}] coordinates{"

# for i in range(len(pointsa)):
#     if pointsa[i][2]==pointsb[i][2]:
#         count_equal=count_equal+1
#         text=text+str_point2d(pointsa[i])
            
# text=text+"};\n\n"






print(f"count_a: {(float(count_a)/float(count_a+count_b+count_c))}")
print(f"count_b: {float(count_b)/float(count_a+count_b+count_c)}")
print(f"count_c: {float(count_c)/float(count_a+count_b+count_c)}")
# print("Delta Avg: "+ str(totaldelta/float(count_a)))
# print("Delta Avg 2: "+ str(delta/float(sc)))

output_text=""
ft=open(template_file,"r")
for line in ft.readlines():
    if line.find(">>>GRAPHDATA<<<")!=-1:
        output_text=output_text+"\n"+text+"\n"
    else:
        line=line.replace(">>>S1<<<", basename(src1_file).split("-")[1].split("_")[1])
        line=line.replace(">>>S2<<<", basename(src2_file).split("-")[1].split("_")[1])
        line=line.replace(">>>S3<<<", basename(src3_file).split("-")[1].split("_")[1])
        line=line.replace(">>>S1P<<<", str( round((float(count_a)/float(count_a+count_b+count_c))*100,2)))
        line=line.replace(">>>S2P<<<", str( round((float(count_b)/float(count_a+count_b+count_c))*100,2)))
        line=line.replace(">>>S3P<<<", str( round((float(count_c)/float(count_a+count_b+count_c))*100,2)))        
        output_text=output_text+line
        
ft.close()


fo=open(tgt_file,"w")
fo.write(output_text)
fo.close()

