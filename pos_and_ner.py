import pickle
import numpy as np
from keras.models import load_model
# Import the required library
from tkinter import *
from tkinter import ttk

########################### Loading trained model ###################################
try:
    pos_model
    ner_model
except:
    pos_model = load_model('POS_model.h5')
    ner_model = load_model('NER_model.h5')

def predict(sentence, model, vocab, tag_map):
    s = [vocab[token] if token in vocab else vocab['<UNK>'] for token in sentence.split(' ')]
    len_s = len(s)
    if len_s<128:
        padding = [vocab["<PAD>"]]*(128-len_s)
        s = s + padding
    else:
        s = s[:128]
    batch_data = np.ones((1, len(s)))
    batch_data[0][:] = s
    sentence = np.array(batch_data).astype(int)
    output = model(sentence)
    outputs = np.argmax(output, axis=2)
    labels = list(tag_map.keys())
    pred = []
    for i in range(len(outputs[0])):
        idx = outputs[0][i] 
        pred_label = labels[idx]
        pred.append(pred_label)
    return pred

try:
    pos_vocab
    pos_tags_map
except:
    with open('pos_vocab.pickle', 'rb') as handle:
        pos_vocab = pickle.load(handle)
    with open('pos_tags_map.pickle', 'rb') as handle:
        pos_tags_map = pickle.load(handle)
    

with open('ner_vocab.pickle', 'rb') as handle:
    ner_vocab = pickle.load(handle)
with open('ner_tags_map.pickle', 'rb') as handle:
    ner_tags_map = pickle.load(handle)
    
    

########################### tkinter GUI ###################################
bg_clr = "#F7F7F7"
win=Tk()

win.geometry("800x450")


win.configure(background=bg_clr)
win.title("POS Tagging and Named Entity Recognition")
title = Label(win,text="POS Tagging and Named Entity Recognition", bg="gray",width="300",height="2",fg="White",font = ("Calibri 16 bold italic")).pack()

r_v=IntVar()

r_v.set(1)
my_data=r_v.get()
r1=Radiobutton(win,text='Parts of speech tagging',value=1,variable=r_v,font=16, bg=bg_clr)
r1.pack()

r2=Radiobutton(win,text='Named entity recognition',value=2,variable=r_v,font=16, bg=bg_clr)
r2.pack()


def get_input():
    my_data=r_v.get()
    if my_data == 1:
        out = ''
        i = 0
        pos_pred = predict(text.get(1.0, "end-1c"), pos_model, pos_vocab, pos_tags_map)
        for x,y in zip(text.get(1.0, "end-1c").split(' '), pos_pred):
            i += 1
            if y != 'O' and i%5 != 0:
                out += f'{x}-->{y}; '
            elif y != 'O' and i%5 == 0:
                out += f'{x}-->{y} \n'
        label.config(text=out, bg = 'gray')
    else:
        out = ''
        i = 0
        ner_pred = predict(text.get(1.0, "end-1c"), ner_model, ner_vocab, ner_tags_map)
        for x,y in zip(text.get(1.0, "end-1c").split(' '), ner_pred):
            i += 1
            if y != 'O' and i%5 != 0:
                out += f'{x}-->{y}; '
            elif y != 'O' and i%3 == 0:
                out += f'{x}-->{y} \n'
        label.config(text=out, bg = 'gray')
        
text=Text(win, width=70, height=2, bg='#EEEEEE', font=('Arial 10'))
text.placeholder = 'Enter a sentence'
text.insert(END, "")
text.pack(padx= 10, pady=10)

btn = Button(win, 
                bg='#000000',
                fg='#b7f731',
                relief='flat',
                text='Execute',
                font=('Arial 10'),
                padx=5,
                pady=5,
                command=get_input)
btn.pack(pady=10)

label=Label(win, text="", font=('Arial 11'), bg = bg_clr)
label.pack()


win.mainloop()