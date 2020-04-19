from tkinter import *
import time
import random
from Config import yinbiao, word, alpha_delay, beta_delay,max_num,folder_name, if_test, express
import pickle

def randomlize(mylist, max_num):
    alpha_list = range(0,len(mylist))
    bei = max_num // len(mylist) + 1
    alpha_list = list(alpha_list) * bei
    #random.shuffle(alpha_list)
    return alpha_list

alpha_num, beta_num, express_num = 0, 0, 0
alpha_list = randomlize(yinbiao,max_num)
beta_list = randomlize(word,max_num)
express_list = randomlize(express,max_num)

def beta(var, f, root, record):
    global beta_num, beta_list
    speak = word[beta_num]
    var.set(speak)
    print(speak)
    beta_num += 1
    if beta_num < len(word) + 1: f.after(beta_delay[beta_list[beta_num - 1]] + 500, lambda : beta(var, f, root, record))
    else:
        time.sleep(4)
        root.destroy()


def alpha(var, f, root, record):
    global alpha_num, alpha_list
    speak = yinbiao[alpha_list[alpha_num]]
    if alpha_num > len(alpha_list)//2: speak = "close_eye and " + speak
    var.set(speak)
    record[0].append(speak)
    alpha_num += 1
    if alpha_num < len(alpha_list): f.after(alpha_delay, lambda : alpha(var, f, root, record))
    else: f.after(alpha_delay, lambda : beta(var, f, root, record))

def natural(var,f,root,record):
    global express_num
    var.set(express[0])
    f.after(3000, lambda : emoji(var, f, root, record))

def emoji(var,f,root,record):
    global express_num
    express_num += 1
    if express_num >= len(express): 
        root.destroy()
        return   
    var.set(express[express_num])
    f.after(4000, lambda : natural(var, f, root, record))


def main():
    root = Tk()
    record = ([], [], [])
    f = Frame(root, height=200, width=200)
    var = StringVar()
    var.set("Let's begin!")
    text_label = Label(f, textvariable=var, bg="white", font=("Arial", 36), width=40, height=10)
    text_label.pack(side=TOP)


    f.pack()
    #f.after(3000, lambda : natural(var, f, root, record))
    f.after(4000, lambda : beta(var, f, root, record))
    #else: f.after(1000, lambda : alpha(var, f, root, record))

    mainloop()
    print("saving record")
    #pickle.dump(record, open(folder_name[0] + "/record.pkl", "wb"))

if __name__ == "__main__":
    main()