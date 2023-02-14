from predict import *
from tkinter import *

w2dic, model = loadmodel()

    # predict(w2dic, model, text)

window = Tk()
window.title("情感分析")
window.geometry("600x350")

entry_value = IntVar()  # IntVar 是tkinter的一个类，可以管理单选按钮
entry_value2 = IntVar()  # IntVar 是tkinter的一个类，可以管理单选按钮

def submit():
    text = message_entry.get('0.0', 'end')
    message_entry2.delete('1.0', 'end')
    res = predict(w2dic, model, text)
    message_entry2.insert('insert', res)
##文本输入1
message_entry = Text(window,show=None,selectbackground="red",relief = GROOVE)
message_entry.place(x=40,y=100,width=260, height=150)
#显示

entry = Label(window, text="请输入文本：")
entry.place(x=40,y=70,width=100, height=30)
##文本显示输入2
message_entry2 = Text(window,show=None,selectbackground="red",relief = GROOVE)
message_entry2.place(x=300,y=100,width=240, height=150)

#翻译按钮
button = Button(window, text="分析",command = submit)
button.place(x=300,y=70,width=60, height=30) # Displaying the button


window.mainloop()  # 消息循环